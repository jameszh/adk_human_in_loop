"""FastAPI chat API with ADK agent backend and Telegram human-in-the-loop approval."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

import agent as agent_module

load_dotenv(override=True)

APP_NAME = "human_in_loop_api"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_API_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

session_service = InMemorySessionService()

# session_id -> Runner
_runners: dict[str, Runner] = {}

# ticket_id -> approval state
_pending: dict[str, dict] = {}

# tracks the last Telegram update offset for long polling
_telegram_offset: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_runner(session_id: str) -> Runner:
    if session_id not in _runners:
        _runners[session_id] = Runner(
            agent=agent_module.root_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
    return _runners[session_id]


async def _ensure_session(session_id: str, user_id: str) -> None:
    existing = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if existing is None:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )


async def _telegram_post(method: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{TELEGRAM_URL}/{method}", json=payload)
        return resp.json()


async def _send_approval_request(ticket_id: str, purpose: str, amount: float) -> Union[int, None]:
    text = (
        f"🔔 *Reimbursement Approval Required*\n\n"
        f"*Purpose:* {purpose}\n"
        f"*Amount:* ${amount:.2f}\n"
        f"*Ticket:* `{ticket_id}`\n\n"
        "Please approve or reject:"
    )
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"approve:{ticket_id}"},
            {"text": "❌ Reject", "callback_data": f"reject:{ticket_id}"},
        ]]
    }
    result = await _telegram_post("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": keyboard,
    })
    if result.get("ok"):
        return result["result"]["message_id"]
    return None


async def _edit_approval_message(chat_id: int, message_id: int, ticket_id: str, decision: str, result_text: str) -> None:
    emoji = "✅" if decision == "approved" else "❌"
    text = (
        f"{emoji} *{decision.upper()}*\n\n"
        f"*Ticket:* `{ticket_id}`\n"
        f"*Agent response:* {result_text}"
    )
    await _telegram_post("editMessageText", {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
    })


async def _answer_callback(callback_query_id: str, text: str) -> None:
    await _telegram_post("answerCallbackQuery", {
        "callback_query_id": callback_query_id,
        "text": text,
    })


async def _handle_callback_query(callback_query: dict) -> None:
    """Process an approve/reject callback from Telegram."""
    global _telegram_offset

    callback_data: str = callback_query.get("data", "")
    callback_query_id: str = callback_query["id"]
    chat_id: int = callback_query["message"]["chat"]["id"]
    message_id: int = callback_query["message"]["message_id"]

    if ":" not in callback_data:
        return

    action, ticket_id = callback_data.split(":", 1)

    if ticket_id not in _pending:
        await _answer_callback(callback_query_id, "Ticket not found or already processed.")
        return

    ticket = _pending[ticket_id]
    if ticket["status"] != "pending":
        await _answer_callback(callback_query_id, "Already processed.")
        return

    decision = "approved" if action == "approve" else "rejected"
    ticket["status"] = "processing"

    updated_part = types.Part(
        function_response=types.FunctionResponse(
            id=ticket["function_call_id"],
            name=ticket["function_call_name"],
            response={
                "status": decision,
                "ticketId": ticket_id,
                "approver_feedback": f"{decision.capitalize()} via Telegram",
            },
        )
    )

    runner = _get_runner(ticket["session_id"])
    text_parts: list[str] = []

    async for event in runner.run_async(
        session_id=ticket["session_id"],
        user_id=ticket["user_id"],
        new_message=types.Content(parts=[updated_part], role="user"),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    text_parts.append(part.text.strip())

    result_text = " ".join(text_parts) if text_parts else f"Request {decision}."
    ticket["status"] = decision
    ticket["result"] = result_text

    await _edit_approval_message(chat_id, message_id, ticket_id, decision, result_text)
    await _answer_callback(callback_query_id, f"Request {decision}!")


async def _telegram_polling_loop() -> None:
    """Background task: long-poll Telegram for callback queries."""
    global _telegram_offset
    print("Telegram long-polling started.")
    while True:
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                resp = await client.post(f"{TELEGRAM_URL}/getUpdates", json={
                    "offset": _telegram_offset,
                    "timeout": 30,
                    "allowed_updates": ["callback_query"],
                })
                data = resp.json()

            if data.get("ok"):
                for update in data.get("result", []):
                    _telegram_offset = update["update_id"] + 1
                    if "callback_query" in update:
                        await _handle_callback_query(update["callback_query"])
        except asyncio.CancelledError:
            print("Telegram polling stopped.")
            break
        except Exception as e:
            print(f"Telegram polling error: {e}")
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# App lifespan — starts/stops the polling loop
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_telegram_polling_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="ADK Chat API with Human-in-the-Loop", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    user_id: str = "default_user"
    message: str


class ChatResponse(BaseModel):
    status: str  # "completed" | "pending_approval"
    response: str
    ticket_id: Union[str, None] = None


class StatusResponse(BaseModel):
    ticket_id: str
    status: str  # "pending" | "approved" | "rejected" | "processing"
    result: Union[str, None] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to the agent. If human approval is required, returns pending status."""
    await _ensure_session(req.session_id, req.user_id)
    runner = _get_runner(req.session_id)

    content = types.Content(role="user", parts=[types.Part(text=req.message)])

    long_running_fc: Union[types.FunctionCall, None] = None
    initial_response: Union[types.FunctionResponse, None] = None
    ticket_id: Union[str, None] = None
    text_parts: list[str] = []

    async for event in runner.run_async(
        session_id=req.session_id,
        user_id=req.user_id,
        new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.text:
                text_parts.append(part.text.strip())
            if part.function_call and part.function_call.id in (event.long_running_tool_ids or []):
                long_running_fc = part.function_call
            if (
                part.function_response
                and long_running_fc
                and part.function_response.id == long_running_fc.id
            ):
                initial_response = part.function_response
                if initial_response.response:
                    ticket_id = initial_response.response.get("ticketId")

    # Long-running tool triggered and returned "pending"
    if (
        long_running_fc
        and initial_response
        and initial_response.response
        and initial_response.response.get("status") == "pending"
    ):
        args = long_running_fc.args or {}
        purpose = args.get("purpose", "Unknown")
        amount = float(args.get("amount", 0))

        await _send_approval_request(ticket_id, purpose, amount)

        _pending[ticket_id] = {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "function_call_id": long_running_fc.id,
            "function_call_name": long_running_fc.name,
            "status": "pending",
            "result": None,
        }

        agent_text = " ".join(text_parts) if text_parts else "Approval request sent to manager."
        return ChatResponse(
            status="pending_approval",
            response=agent_text,
            ticket_id=ticket_id,
        )

    return ChatResponse(
        status="completed",
        response=" ".join(text_parts) if text_parts else "Done.",
    )


@app.get("/status/{ticket_id}", response_model=StatusResponse)
async def get_status(ticket_id: str):
    """Poll for the result of a pending approval."""
    if ticket_id not in _pending:
        raise HTTPException(status_code=404, detail="Ticket not found")
    ticket = _pending[ticket_id]
    return StatusResponse(
        ticket_id=ticket_id,
        status=ticket["status"],
        result=ticket.get("result"),
    )
