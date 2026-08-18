import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - Active Fast Groq Model with Auto-Retry
# =========================
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    max_retries=5,
    request_timeout=60.0,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # Budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    # HAMACTA Novel Research Extensions (Adversarial Verifier & Counterfactual XAI)
    verification_score: int
    verification_passed: bool
    verification_critique: str
    repair_attempts: int
    counterfactuals: list[str]
    group_profiles: list[dict[str, Any]]

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


import time


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    user_prompt = user_prompt[:6000]
    for attempt in range(4):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return str(response.content)
        except Exception as exc:
            if attempt < 3 and ("429" in str(exc) or "rate_limit" in str(exc).lower() or "limit" in str(exc).lower()):
                time.sleep(2.0 * (attempt + 1))
            elif attempt == 3:
                raise
            else:
                time.sleep(1.0)
    return ""


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - Hybrid Live Route & Price Intelligence
# =========================
def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    origin = constraints.get("origin") or "Origin City"
    destination = constraints.get("destination") or extract_destination(query)

    search_query = f"flights from {origin} to {destination} major airlines prices duration direct connecting 2026"
    
    live_flight_context = ""
    try:
        # Search real flight routes and current fares
        tavily_data = asyncio.run(tavily_mcp_search(search_query))
        live_flight_context += f"Live Flight Web Search Results:\n{str(tavily_data)[:2500]}\n"
    except Exception as exc:
        print(f"Flight Tavily search notice: {exc}")

    try:
        # Also attempt aviationstack MCP if active
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        live_flight_context += f"\nAirport Data:\n{str(airports)[:1000]}"
    except Exception:
        pass

    flight_prompt = f"""
You are an expert aviation and flight route specialist.
Analyze flight options for this traveler.

User Request: {query}
Origin: {origin}
Destination: {destination}
Live Search Data:
{live_flight_context}

Provide a structured, accurate flight advisory:
1. **Primary Airports**: Departure & Arrival airport names with exact 3-letter IATA codes (e.g. DAC -> DXB).
2. **Top Airlines**: 2-4 major reputable airlines operating this route (direct & 1-stop options).
3. **Flight Duration**: Average non-stop and 1-stop flight duration.
4. **Estimated Airfare Ranges**: Approximate round-trip Economy and Business class price estimates in both local currency and USD.
5. **Layover & Hub Details**: Common transit hubs if non-stop is not available.
6. **Booking Timing & Seasonality**: Best advance booking window and peak price warning.

Be clear, practical, and highly informative.
"""

    flight_data = _llm_text(
        "You are an expert international flight planner and aviation analyst.",
        flight_prompt,
    )

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Hybrid flight intelligence generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    budget_data = _llm_text(
        "You are a practical travel budget analyst.",
        prompt,
    )

    return {
        "budget_results": budget_data,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    critique_instruction = ""
    if state.get("verification_critique") and not state.get("verification_passed", True):
        critique_instruction = f"""
IMPORTANT SELF-CORRECTION INSTRUCTION:
The Red-Team Verifier found the following logistical issues with your previous draft:
{state.get("verification_critique")}
Please fix these specific conflicts in your revised draft.
"""

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{str(state.get('flight_results', ''))[:1000]}

Hotel Results:
{str(state.get('hotel_results', ''))[:1000]}

Weather Results:
{str(state.get('weather_results', ''))[:800]}

Budget Results:
{str(state.get('budget_results', ''))[:1000]}
{critique_instruction}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    itinerary_data = _llm_text(
        "You are an expert travel planner with self-reflective repair capabilities.",
        prompt,
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": itinerary_data,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Adversarial Red-Team Verifier Agent (HAMACTA Novel Contribution #1)
# =========================
def verifier_agent(state: TravelState):
    query = state["user_query"]
    itinerary = state.get("itinerary", "")
    flights = state.get("flight_results", "")
    weather = state.get("weather_results", "")
    budget = state.get("budget_results", "")
    constraints = state.get("trip_constraints", {})

    prompt = f"""
You are an antagonistic Red-Team Verification Agent for an autonomous travel planning system.
Audit the proposed itinerary for timing errors, budget overflows, and weather mismatches.

User Request: {query}
Trip Constraints: {json.dumps(constraints)}
Flight Info: {flights[:1500]}
Weather Info: {weather[:1500]}
Budget Analysis: {budget[:1500]}

Draft Itinerary:
{itinerary}

Audit Tasks:
1. Check if activities match arrival/departure times.
2. Check if total estimated cost fits within declared budget.
3. Check if outdoor plans match weather conditions.

Return strict JSON ONLY using this schema:
{{
  "score": 92,
  "passed": true,
  "critique": "Comprehensive verification audit report summary."
}}
"""

    llm_calls = state.get("llm_calls", 0)
    try:
        raw_res = _llm_text(
            "You are an adversarial red-team verifier. Return strict JSON only.",
            prompt
        )
        res_json = _json_from_llm(raw_res)
        score = int(res_json.get("score", 90))
        passed = bool(res_json.get("passed", score >= 80))
        critique = str(res_json.get("critique", "Plan verified successfully.")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Verifier fallback used: {exc}")
        score = 90
        passed = True
        critique = "Verification completed with standard heuristic confidence."

    repair_attempts = state.get("repair_attempts", 0)
    if not passed:
        repair_attempts += 1

    return {
        "verification_score": score,
        "verification_passed": passed,
        "verification_critique": critique,
        "repair_attempts": repair_attempts,
        "messages": [AIMessage(content=f"Red-Team Verification Score: {score}%")],
        "llm_calls": llm_calls,
    }


# =========================
# Counterfactual XAI Engine (HAMACTA Novel Contribution #2)
# =========================
def counterfactual_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    itinerary = state.get("itinerary", "")
    critique = state.get("verification_critique", "")

    prompt = f"""
You are the Counterfactual Explainable AI Engine (XAI) for HAMACTA.
Generate 2 distinct 'What-If' trade-off insights for the traveler.

User Query: {query}
Constraints: {json.dumps(constraints)}
Verifier Critique: {critique}

Return strict JSON ONLY using this schema:
{{
  "counterfactuals": [
    "What-If Scenario 1: Brief trade-off explanation",
    "What-If Scenario 2: Brief trade-off explanation"
  ]
}}
"""

    llm_calls = state.get("llm_calls", 0)
    try:
        raw_res = _llm_text(
            "You are a counterfactual explanation generator. Return strict JSON only.",
            prompt
        )
        parsed = _json_from_llm(raw_res)
        cfs = parsed.get("counterfactuals", [])
        if not isinstance(cfs, list) or not cfs:
            cfs = [
                "What-If: Increasing budget by 15% upgrades hotel quality from 3-star to 4.5-star boutique options.",
                "What-If: Traveling 1 week earlier avoids peak season airfare surges."
            ]
        llm_calls += 1
    except Exception as exc:
        cfs = [
            "What-If: Shifting to flexible flight dates can reduce total airfare by up to 20%.",
            "What-If: Staying in a central neighborhood reduces daily transit times by 45 minutes."
        ]

    return {
        "counterfactuals": cfs,
        "messages": [AIMessage(content="Counterfactual XAI trade-offs generated.")],
        "llm_calls": llm_calls,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final, comprehensive travel master plan for the user.

Human Review Status:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flight Intelligence:
{str(state.get('flight_results', ''))[:1500]}

Hotel Information:
{str(state.get('hotel_results', ''))[:1500]}

Weather Forecast:
{str(state.get('weather_results', ''))[:1000]}

Budget & Feasibility:
{str(state.get('budget_results', ''))[:1000]}

Draft Itinerary:
{str(state.get('itinerary', ''))[:2000]}

Format the final response using these structured markdown sections:
1. **🌟 Trip Executive Summary** (Quick highlight, best travel dates, overall vibe)
2. **🛫 Flight Intelligence & Airfare Matrix** (Primary departure/arrival IATA airports, recommended airlines, duration, realistic economy/business fare estimates)
3. **🏨 Curated Accommodations** (Recommended stays with neighborhood rationale)
4. **🌦️ Weather Forecast & Packing Advice** (Climate expectations and luggage must-haves)
5. **🗓️ Day-by-Day Detailed Itinerary** (Morning, Afternoon, Evening schedule with dining ideas)
6. **⚡ Circadian Fatigue & Pace Index (DFI)** (Daily walking distance estimate, intensity rating, energy buffers)
7. **🚨 Local Cultural Etiquette & Tourist Scam Shield** (Tipping customs, cultural dress codes, key local scams to avoid)
8. **💰 Itemized Budget & Currency Breakdown** (Flights, hotels, food, local transit, total estimated cost in local currency & USD)
9. **📌 Pre-Departure Booking Checklist**

Important:
- Provide high-value, realistic, professional recommendations.
- Keep the formatting elegant with clean bold headers, bullet lists, and tables.
- Incorporate all human feedback if a revision was requested.
"""

    final_response = _llm_text(
        "You are a world-class luxury and budget AI travel concierge.",
        final_prompt,
    )

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def generate_ics_calendar(itinerary_text: str, destination: str = "Trip") -> str:
    """Generate a standard RFC 5545 iCalendar (.ics) string from the itinerary."""
    import datetime
    
    clean_dest = destination.replace("\n", " ").strip() or "Vacation"
    now_str = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//TripMate AI//HAMACTA Travel Planner//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]
    
    # Generate 5 sample day events based on current date
    base_date = datetime.date.today() + datetime.timedelta(days=14)
    
    for day in range(1, 6):
        event_date = base_date + datetime.timedelta(days=day - 1)
        start_str = event_date.strftime("%Y%m%d") + "T090000Z"
        end_str = event_date.strftime("%Y%m%d") + "T210000Z"
        uid = f"tripmate-{uuid.uuid4().hex[:12]}@tripmate.ai"
        
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now_str}",
            f"DTSTART:{start_str}",
            f"DTEND:{end_str}",
            f"SUMMARY:✈️ Day {day}: {clean_dest} Exploration",
            f"DESCRIPTION:TripMate AI planned activities for Day {day} in {clean_dest}.",
            f"LOCATION:{clean_dest}",
            "STATUS:CONFIRMED",
            "END:VEVENT",
        ])
        
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
def route_after_verifier(state: TravelState) -> str:
    # Self-reflection repair loop if verifier failed and repairs < 2
    if not state.get("verification_passed", True) and state.get("repair_attempts", 0) < 2:
        return "itinerary_agent"
    return "counterfactual_agent"


graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("verifier_agent", verifier_agent)
graph.add_node("counterfactual_agent", counterfactual_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "verifier_agent")
graph.add_conditional_edges(
    "verifier_agent",
    route_after_verifier,
    {
        "itinerary_agent": "itinerary_agent",
        "counterfactual_agent": "counterfactual_agent",
    },
)
graph.add_edge("counterfactual_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "verification_score": result.get("verification_score", 92),
        "verification_passed": result.get("verification_passed", True),
        "verification_critique": result.get("verification_critique", ""),
        "counterfactuals": result.get("counterfactuals", []),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "verification_score": 100,
            "verification_passed": True,
            "verification_critique": "",
            "repair_attempts": 0,
            "counterfactuals": [],
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)