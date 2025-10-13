import os
import uuid
import json
import threading
from time import perf_counter
from datetime import datetime, timedelta
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

from src.planning_agent import planner_agent, executor_agent_step

import html, textwrap

# === Load env vars ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Fix for Heroku's postgres:// URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")


# === DB setup ===
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)


try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"\u274c DB creation failed: {e}")

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"\u274c DB creation failed: {e}")

# === FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

task_progress = {}


class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    return {"status": "ok"}


@app.post("/generate_report")
def generate_report(req: PromptRequest):
    task_id = str(uuid.uuid4())
    db = SessionLocal()
    db.add(Task(id=task_id, prompt=req.prompt, status="running"))
    db.commit()
    db.close()

    task_progress[task_id] = {"steps": []}
    initial_plan_steps = planner_agent(req.prompt, model=req.model)
    for step_title in initial_plan_steps:
        task_progress[task_id]["steps"].append(
            {
                "title": step_title,
                "status": "pending",
                "description": "Awaiting execution",
                "substeps": [],
            }
        )

    thread = threading.Thread(
        target=run_agent_workflow, args=(task_id, req.prompt, initial_plan_steps, req.model)
    )
    thread.start()
    return {"task_id": task_id}


@app.get("/task_progress/{task_id}")
def get_task_progress(task_id: str):
    return task_progress.get(task_id, {"steps": []})


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": task.status,
        "result": json.loads(task.result) if task.result else None,
    }


def format_history(history):
    return "\n\n".join(
        f"🔹 {title}\n{desc}\n\n📝 Output:\n{output}" for title, desc, output in history
    )


def run_agent_workflow(task_id: str, prompt: str, initial_plan_steps: list, model: Optional[str] = None):
    steps_data = task_progress[task_id]["steps"]
    execution_history = []
    step_start_times: dict[int, float] = {}
    shared_state: dict = {}

    def update_step_status(index, status, description="", substep=None, started_at=None, duration=None):
        if index < len(steps_data):
            steps_data[index]["status"] = status
            if description:
                steps_data[index]["description"] = description
            if substep:
                steps_data[index]["substeps"].append(substep)
            steps_data[index]["updated_at"] = datetime.utcnow().isoformat()
            if started_at:
                steps_data[index]["started_at"] = started_at
            if duration is not None:
                steps_data[index]["duration_seconds"] = round(duration, 2)
                steps_data[index]["duration_human"] = str(timedelta(seconds=round(duration)))

    try:
        for i, plan_step_title in enumerate(initial_plan_steps):
            step_start_times[i] = perf_counter()
            started_at_iso = datetime.utcnow().isoformat()
            update_step_status(
                i,
                "running",
                f"Executing: {plan_step_title}",
                started_at=started_at_iso,
            )

            actual_step_description, agent_name, output = executor_agent_step(
                plan_step_title,
                execution_history,
                prompt,
                model=model,
                shared_state=shared_state,
            )

            execution_history.append([plan_step_title, actual_step_description, output])

            def esc(s: str) -> str:
                return html.escape(s or "")

            def nl2br(s: str) -> str:
                return esc(s).replace("\n", "<br>")

            # ...
            start_ts = step_start_times.pop(i, None)
            elapsed = perf_counter() - start_ts if start_ts is not None else 0.0

            update_step_status(
                i,
                "done",
                f"Completed: {plan_step_title}",
                {
                    "title": f"Called {agent_name}",
                    "content": f"""
<div style='border:1px solid #ccc; border-radius:8px; padding:10px; margin:8px 0; background:#fff;'>
  <div style='font-weight:bold; color:#2563eb;'>📘 User Prompt</div>
  <div style='white-space:pre-wrap;'>{prompt}</div>

  <div style='font-weight:bold; color:#16a34a; margin-top:8px;'>📜 Previous Step</div>
  <pre style='white-space:pre-wrap; background:#f9fafb; padding:6px; border-radius:6px; margin:0;'>
{format_history(execution_history[-2:-1])}
  </pre>

  <div style='font-weight:bold; color:#f59e0b; margin-top:8px;'>🧹 Your next task</div>
  <div style='white-space:pre-wrap;'>{actual_step_description}</div>

  <div style='font-weight:bold; color:#10b981; margin-top:8px;'>✅ Output</div>
  <!-- ⚠️ NO <pre> AQUÍ -->
  <div style='white-space:pre-wrap;'>
{output}
  </div>
</div>
""".strip(),
                },
                duration=elapsed,
            )

        final_report_markdown = (
            execution_history[-1][-1] if execution_history else "No report generated."
        )

        result = {"html_report": final_report_markdown, "history": steps_data}

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "done"
        task.result = json.dumps(result)
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()

    except Exception as e:
        print(f"Workflow error for task {task_id}: {e}")
        if steps_data:
            error_step_index = next(
                (i for i, s in enumerate(steps_data) if s["status"] == "running"),
                len(steps_data) - 1,
            )
            if error_step_index >= 0:
                start_ts = step_start_times.pop(error_step_index, None)
                elapsed = perf_counter() - start_ts if start_ts is not None else 0.0
                update_step_status(
                    error_step_index,
                    "error",
                    f"Error during execution: {e}",
                    {"title": "Error", "content": str(e)},
                    duration=elapsed,
                )

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "error"
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()
