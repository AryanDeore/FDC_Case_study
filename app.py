import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Load artifacts ---
qr_lower = joblib.load('model_artifacts/qr_lower.pkl')
qr_median = joblib.load('model_artifacts/qr_median.pkl')
qr_upper = joblib.load('model_artifacts/qr_upper.pkl')
scaler = joblib.load('model_artifacts/scaler.pkl')
te_company = joblib.load('model_artifacts/te_company.pkl')
te_sub_area = joblib.load('model_artifacts/te_sub_area.pkl')
tfidf = joblib.load('model_artifacts/tfidf.pkl')
expensive_centroid = joblib.load('model_artifacts/expensive_centroid.pkl')
cheap_centroid = joblib.load('model_artifacts/cheap_centroid.pkl')
q_hat = joblib.load('model_artifacts/q_hat.pkl')

# --- Dropdown options ---
COMPANIES = [
    'ace constructions', 'anp corp', 'bhaktamar realities', 'calyx spaces',
    'dolphin group', 'godrej properties', 'jhala group', 'kalpataru',
    'keystone landmark', 'kohinoor group', 'kundan spaces', 'lush life',
    'maha anand pinnac associates', 'majestique landmarks', 'mantra properties',
    'nirman developers', 'oxy buildcorp', 'porwal  develkoper',
    'porwal & anand develkoper', 'proviso group', 'puraniks', 'ravima ventures',
    'sagitarius ecospaces llp', 'shapoorji paloonji', 'shroff developers',
    'sukwani associates', 'supreme', 'tejraaj group', 'unique properties',
    'urban space creator', 'vasupujya corporation', 'venkatesh bhoomi construction',
    'vijaya laxmi creations', 'vijaya laxmi infrarealtors', 'vtp reality',
    'waghvani constructions', 'wellwisher apartments'
]

SUB_AREAS = [
    'akurdi', 'balewadi', 'baner', 'bavdhan', 'bavdhan budruk', 'bt kawade rd',
    'dhanori', 'hadapsar', 'handewadi', 'hinjewadi', 'karvanagar', 'kayani nagar',
    'keshav nagar', 'kharadi', 'kirkatwadi sinhagad road', 'kiwale',
    'koregaon park', 'koregoan', 'lonavala', 'magarpatta', 'mahalunge', 'manjri',
    'mohammadwadi', 'mundhwa', 'nibm', 'pisoli', 'ravet', 'susgaon', 'talegoan',
    'tathawade', 'undri', 'vimannagar', 'wadgaon sheri'
]


def compute_desc_price_signal(description):
    """Compute description price signal using TF-IDF similarity."""
    if not description or description.strip() == '':
        return 0.0
    tfidf_vec = tfidf.transform([description])
    sim_expensive = cosine_similarity(tfidf_vec, expensive_centroid.reshape(1, -1))[0][0]
    sim_cheap = cosine_similarity(tfidf_vec, cheap_centroid.reshape(1, -1))[0][0]
    return sim_expensive - sim_cheap


def predict(area_sqft, bedrooms, swimming_pool, mall, hospital, school,
            clubhouse, park_jogging_track, gym, company_name, sub_area,
            description, listing_price=None):
    """Run feature pipeline + inference, return price range and verdict."""

    # Binary amenities
    pool = 1 if swimming_pool else 0
    neighbourhood_amenities = sum([
        1 if mall else 0,
        1 if hospital else 0,
        1 if school else 0,
    ])
    property_amenities = sum([
        1 if clubhouse else 0,
        1 if park_jogging_track else 0,
        1 if gym else 0,
    ])

    # Description signal
    desc_signal = compute_desc_price_signal(description)

    # Log transform area
    area_log = np.log1p(area_sqft)

    # Target encode company and sub_area
    company_encoded = te_company.transform(
        np.array([[company_name]])
    )[0][0]
    sub_area_encoded = te_sub_area.transform(
        np.array([[sub_area]])
    )[0][0]

    # Build feature vector (same column order as training)
    features = pd.DataFrame([{
        'swimming_pool': pool,
        'no_of_bedrooms': bedrooms,
        'neighbourhood_amenities': neighbourhood_amenities,
        'property_amenities': property_amenities,
        'desc_price_signal': desc_signal,
        'property_area_log': area_log,
        'te_company_name': company_encoded,
        'te_sub_area': sub_area_encoded,
    }])

    # Scale (keep as DataFrame to preserve feature names)
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # Inference
    pred_lower = np.expm1(qr_lower.predict(features_scaled)[0] - q_hat)
    pred_median = np.expm1(qr_median.predict(features_scaled)[0])
    pred_upper = np.expm1(qr_upper.predict(features_scaled)[0] + q_hat)

    # Build result HTML
    result = (
        f"<div class='result-range'>{pred_lower:.1f} &ndash; {pred_upper:.1f} lakhs</div>"
        f"<div class='result-median'>Predicted price: {pred_median:.1f} lakhs</div>"
    )

    # Range bar (always shown)
    bar_min = pred_lower * 0.85
    bar_max = pred_upper * 1.15
    bar_span = bar_max - bar_min
    lower_pct = ((pred_lower - bar_min) / bar_span) * 100
    upper_pct = ((pred_upper - bar_min) / bar_span) * 100
    median_pct = ((pred_median - bar_min) / bar_span) * 100

    range_bar = (
        f"<div class='range-bar-container'>"
        f"<div class='range-bar-track'>"
        f"<div class='range-bar-fill' style='left: {lower_pct:.1f}%; width: {upper_pct - lower_pct:.1f}%;'></div>"
        f"<div class='range-bar-median' style='left: {median_pct:.1f}%;'></div>"
    )

    if listing_price is not None and listing_price > 0:
        marker_pct = max(0, min(100, ((listing_price - bar_min) / bar_span) * 100))
        range_bar += f"<div class='range-bar-marker' style='left: {marker_pct:.1f}%;'></div>"

    range_bar += (
        f"</div>"
        f"<div class='range-bar-labels'>"
        f"<span>{pred_lower:.1f}</span>"
        f"<span style='position: absolute; left: {median_pct:.1f}%; transform: translateX(-50%);'>{pred_median:.1f}</span>"
        f"<span>{pred_upper:.1f}</span>"
        f"</div>"
        f"</div>"
    )
    result += range_bar

    if listing_price is not None and listing_price > 0:
        range_width = pred_upper - pred_lower
        if listing_price < pred_lower:
            result += f"<div class='verdict verdict-under'>Below Range &mdash; {listing_price:.1f} lakhs is below the estimated range</div>"
        elif listing_price > pred_upper:
            result += f"<div class='verdict verdict-over'>Above Range &mdash; {listing_price:.1f} lakhs is above the estimated range</div>"
        elif listing_price < pred_lower + range_width * 0.25:
            result += f"<div class='verdict verdict-low'>Near Lower End &mdash; {listing_price:.1f} lakhs, in the lower quarter of the range</div>"
        elif listing_price > pred_upper - range_width * 0.25:
            result += f"<div class='verdict verdict-high'>Near Upper End &mdash; {listing_price:.1f} lakhs, in the upper quarter of the range</div>"
        else:
            result += f"<div class='verdict verdict-ok'>Within Range &mdash; {listing_price:.1f} lakhs, well within the estimated range</div>"
        result += "<div class='result-basis'>Your price is compared against the model's estimated range.</div>"
    else:
        result += "<div class='result-basis'>Based on area, bedrooms, builder, locality, amenities, and description.</div>"

    return result


# --- Load CSS from file ---
with open(os.path.join(os.path.dirname(__file__), 'style.css')) as f:
    CUSTOM_CSS = f.read()

# --- Gradio UI ---
with gr.Blocks(title="Pillow: Price Discovery for Pune Real Estate") as app:
    gr.HTML("<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>")

    gr.HTML("<h1 style='font-family: Inter, sans-serif; font-size: 2.2rem; font-weight: 700; text-align: center; color: #1d4ed8; margin: 0 0 4px;'>Pillow</h1>")
    gr.HTML("<p style='font-family: Inter, sans-serif; font-size: 1rem; text-align: center; color: #666; margin: 0 0 24px;'>Estimate fair value for Pune real estate</p>")

    with gr.Row(elem_classes=["main-layout"]):
        # Left column: stepped inputs
        with gr.Column(scale=3, elem_classes=["input-column"]):
            with gr.Column(elem_classes=["card-panel"]):
                gr.HTML("<div class='step-header'><span class='step-badge'>1</span><div class='section-title'><svg viewBox='0 0 24 24'><path d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/><polyline points='9 22 9 12 15 12 15 22'/></svg>Property Details</div></div>")
                area_sqft = gr.Slider(200, 3000, value=900, step=1, label="Property Area (sq ft)")
                bedrooms = gr.Slider(1, 6, value=2, step=0.5, label="Number of Bedrooms")
                with gr.Row():
                    company_name = gr.Dropdown(COMPANIES, label="Builder / Company", value='mantra properties', elem_classes=["styled-dropdown"])
                    sub_area = gr.Dropdown(SUB_AREAS, label="Sub Area", value='baner', elem_classes=["styled-dropdown"])

            with gr.Column(elem_classes=["card-panel"]):
                gr.HTML("<div class='step-header'><span class='step-badge'>2</span><div class='section-title'><svg viewBox='0 0 24 24'><rect x='3' y='3' width='7' height='7'/><rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/><rect x='3' y='14' width='7' height='7'/></svg>Amenities</div></div>")
                with gr.Row(equal_height=True, elem_classes=["amenity-row"]):
                    swimming_pool = gr.Checkbox(label="Swimming Pool")
                    clubhouse = gr.Checkbox(label="Clubhouse")
                    park_jogging_track = gr.Checkbox(label="Park / Jogging Track")
                    gym_check = gr.Checkbox(label="Gym")
                with gr.Row(equal_height=True, elem_classes=["amenity-row"]):
                    school = gr.Checkbox(label="School / University")
                    mall = gr.Checkbox(label="Mall in Township")
                    hospital = gr.Checkbox(label="Hospital in Township")
                    select_all = gr.Checkbox(label="Select All")

            with gr.Column(elem_classes=["card-panel"]):
                gr.HTML("<div class='step-header'><span class='step-badge'>3</span><div class='section-title'><svg viewBox='0 0 24 24'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/><line x1='16' y1='17' x2='8' y2='17'/><polyline points='10 9 9 9 8 9'/></svg>Description</div></div>")
                description = gr.Textbox(
                    label="Property Description",
                    placeholder="This is a luxurious property boasting 4.5 bedrooms in a tall building with a river front view.",
                    lines=3,
                )

            with gr.Column(elem_classes=["card-panel"]):
                gr.HTML("<div class='step-header'><span class='step-badge'>4</span><div class='section-title'><svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><polyline points='12 6 12 12 16 14'/></svg>Choose an Action</div></div>")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["action-card"]):
                        gr.HTML("<div class='action-label'>Get the model's estimated price range</div>")
                        gr.HTML("<div class='action-spacer'></div>")
                        estimate_btn = gr.Button("Estimate Range", variant="primary", elem_classes=["gen-btn"])

                    with gr.Column(scale=1, elem_classes=["action-card"]):
                        gr.HTML("<div class='action-label'>Enter your price and see if it's fair</div>")
                        listing_price = gr.Number(
                            label="Your Price (in lakhs)",
                            value=None,
                            minimum=0,
                        )
                        check_btn = gr.Button("Check My Price", variant="secondary", elem_classes=["gen-btn", "check-btn"])

        # Right column: sticky result panel
        with gr.Column(scale=2, elem_classes=["result-column"]):
            with gr.Column(elem_classes=["result-panel"]):
                gr.HTML("<div class='section-title' style='justify-content: center;'><svg viewBox='0 0 24 24'><line x1='12' y1='1' x2='12' y2='23'/><path d='M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'/></svg>Estimated Price Range</div>")
                output = gr.HTML(
                    value=(
                        "<div class='empty-state'>"
                        "<svg width='48' height='48' viewBox='0 0 24 24' fill='none' stroke='#b0c4de' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='margin-bottom: 12px;'>"
                        "<path d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/>"
                        "<polyline points='9 22 9 12 15 12 15 22'/>"
                        "</svg>"
                        "<div class='empty-state-text'>Fill in property details and choose an action</div>"
                        "</div>"
                    )
                )

    amenity_checkboxes = [swimming_pool, clubhouse, park_jogging_track, gym_check, school, mall, hospital]

    def toggle_select_all(checked):
        if checked:
            return [gr.update(value=True, interactive=False) for _ in amenity_checkboxes]
        else:
            return [gr.update(value=False, interactive=True) for _ in amenity_checkboxes]

    select_all.change(
        fn=toggle_select_all,
        inputs=[select_all],
        outputs=amenity_checkboxes,
    )

    estimate_btn.click(
        fn=lambda *args: predict(*args, listing_price=None),
        inputs=[area_sqft, bedrooms, swimming_pool, mall, hospital, school,
                clubhouse, park_jogging_track, gym_check, company_name, sub_area,
                description],
        outputs=output,
    )

    check_btn.click(
        fn=predict,
        inputs=[area_sqft, bedrooms, swimming_pool, mall, hospital, school,
                clubhouse, park_jogging_track, gym_check, company_name, sub_area,
                description, listing_price],
        outputs=output,
    )

    gr.HTML(
        "<div class='footer'>"
        "<a href='https://linkedin.com/in/aryandeore' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/></svg>"
        "aryandeore"
        "</a>"
        "<a href='https://www.aryandeore.ai' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm-1 4.062V8H7.062A8.006 8.006 0 0 1 11 4.062zM4.062 13H8v3H5.674A7.953 7.953 0 0 1 4.062 13zm1.612 5H8v2.938A8.006 8.006 0 0 1 5.674 18zM11 19.938V16h3.326A8.006 8.006 0 0 1 11 19.938zM11 14v-3h2v3h-2zm0-5V4.062A8.006 8.006 0 0 1 14.938 8H11zm5 9v-3h2.326a7.953 7.953 0 0 1-2.326 3zm2.326-5H16v-3h3.938a7.953 7.953 0 0 1-1.612 3zM16 8V5.062A8.006 8.006 0 0 1 19.938 9H16zm-3-3.938V8h-2V4.062A8.006 8.006 0 0 1 13 4.062z'/></svg>"
        "aryandeore.ai"
        "</a>"
        "<a href='mailto:aryandeore.work@gmail.com'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.9.732-1.636 1.636-1.636h.749L12 10.638 21.615 3.82h.749c.904 0 1.636.737 1.636 1.636z'/></svg>"
        "Get in touch"
        "</a>"
        "</div>"
    )

if __name__ == '__main__':
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CUSTOM_CSS,
    )
