
import gradio as gr
import os
from dotenv import load_dotenv
from flight_agent.crew import FlightAgent




load_dotenv()


def search_flights(
    origin,
    destination,
    departure_date,
    return_date,
    passengers,
    cabin_class,
):
    if not origin or not destination or not departure_date:
        return "⚠️ Please fill in Origin, Destination and Departure Date."

    try:
        inputs = {
            "origin": origin,
            "destination": destination,
            "departure_date": str(departure_date),
            "return_date": str(return_date) if return_date else "N/A",
            "passengers": str(passengers),
            "cabin_class": cabin_class,
        }

        result = FlightAgent().crew().kickoff(inputs=inputs)
        return str(result)

    except Exception as e:
        return f"❌ Agent error: {str(e)}"


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="✈️ AI Flight Search") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="header">
            <h1>✈️ AI Flight Search Agent</h1>
            <p style="color: #6b7280; font-size: 15px;">
                Powered by CrewAI · Apify Google Flights · Tavily Web Search
            </p>
        </div>
    """)

    # ── Input form ────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛫 Search Details")

            with gr.Row():
                origin = gr.Textbox(
                    label="Origin",
                    placeholder="e.g. Delhi (DEL)",
                    scale=1,
                )
                destination = gr.Textbox(
                    label="Destination",
                    placeholder="e.g. London (LHR)",
                    scale=1,
                )

            with gr.Row():
                departure_date = gr.Textbox(
                    label="Departure Date",
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )
                return_date = gr.Textbox(
                    label="Return Date (optional)",
                    placeholder="YYYY-MM-DD",
                    scale=1,
                )

            with gr.Row():
                passengers = gr.Slider(
                    label="Passengers",
                    minimum=1,
                    maximum=9,
                    value=1,
                    step=1,
                    scale=1,
                )
                cabin_class = gr.Dropdown(
                    label="Cabin Class",
                    choices=["Economy", "Premium Economy", "Business", "First"],
                    value="Economy",
                    scale=1,
                )

            search_btn = gr.Button(
                "🔍 Search Flights",
                variant="primary",
                elem_classes=["search-btn"],
            )

    gr.Markdown("---")

    # ── Output ────────────────────────────────────────────────────────────────
    gr.Markdown("### 📋 Agent Recommendations")

    with gr.Row():
        output = gr.Markdown(
            value="Results will appear here after you search...",
            elem_classes=["output-box"],
        )

    # ── Status indicator ──────────────────────────────────────────────────────
    with gr.Row():
        status = gr.Textbox(
            label="Agent Status",
            value="Idle",
            interactive=False,
            scale=1,
        )

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Markdown("### 💡 Try these examples")
    gr.Examples(
        examples=[
            ["Delhi (DEL)", "London (LHR)", "2025-06-15", "2025-06-25", 1, "Economy"],
            ["Mumbai (BOM)", "New York (JFK)", "2025-07-01", "2025-07-15", 2, "Business"],
            ["Bangalore (BLR)", "Dubai (DXB)", "2025-08-10", "2025-08-17", 1, "Economy"],
        ],
        inputs=[origin, destination, departure_date, return_date, passengers, cabin_class],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div style="text-align:center; padding: 20px 0 5px 0; color: #9ca3af; font-size: 13px;">
            Agent may take 1–3 minutes to complete · Results include live pricing
        </div>
    """)

    # ── Wiring ────────────────────────────────────────────────────────────────
    def run_with_status(origin, destination, departure_date, return_date, passengers, cabin_class):
        yield gr.update(value="⏳ Agent is working..."), gr.update(value="*Searching flights, please wait...*")
        result = search_flights(origin, destination, departure_date, return_date, passengers, cabin_class)
        yield gr.update(value="✅ Done"), gr.update(value=result)

    search_btn.click(
        fn=run_with_status,
        inputs=[origin, destination, departure_date, return_date, passengers, cabin_class],
        outputs=[status, output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )