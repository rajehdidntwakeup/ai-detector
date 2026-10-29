"""Build and validate the initial Phase 2 screening corpus."""
import hashlib
import json
from pathlib import Path
from evaluation.run import load_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("manifest.jsonl")


def write_manifest(records):
    MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


PUBLIC_DOMAIN_HUMAN = [
    {
        "id": "pd-human-nightingale-1", "label": "human", "split": "development",
        "group": "public-domain-human-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "low",
        "text": (
            "The very first requirement in a hospital is that it should do the sick no harm. "
            "It seems a small matter to attend to this; and yet, what is done in the construction "
            "and management of ordinary sick wards, both public and private, often inflicts injury "
            "upon the sick. The want of fresh air, or the want of warmth, or the want of quiet, or "
            "the want of light, or the want of cleanliness, or bad ventilation, are causes of injury "
            "often quite as much as the physician's want of care. It is, in fact, as impossible to "
            "learn sound nursing in a sick room made unhealthy, as it is to learn navigation on board "
            "a ship that will not sail. The nurse must learn to observe, so as to be able to report "
            "exactly the state of the sick. She must be able to distinguish between a change for the "
            "better and a change for the worse, and to judge whether the patient is improving or not."
        ),
        "source": "Florence Nightingale, Notes on Nursing (public domain, 1859/1860)",
        "permission": "Public domain excerpt",
    },
    {
        "id": "pd-human-osler-1", "label": "human", "split": "calibration",
        "group": "public-domain-human-academic-2026-10-cal",
        "domain": "medicine", "citationDensity": "low",
        "text": (
            "Pneumonia may be defined as an acute infectious disease, characterized anatomically by "
            "solidification of the pulmonary tissue due to an exudation into the alveoli, and clinically "
            "by a sudden onset with high temperature, rapid pulse, dyspnoea, cough, and pain in the side. "
            "The anatomical lesion is usually lobar in distribution, though in many cases, particularly "
            "at the extremes of life, it is lobular or broncho-pneumonic. The specific cause is the "
            "pneumococcus, though other organisms, notably the streptococcus and staphylococcus, may "
            "produce a similar lesion. The recognition of the disease is rarely difficult, and the "
            "diagnosis can usually be confirmed by physical signs, by the character of the sputum, and "
            "by the course of the temperature. The prognosis depends chiefly upon the age of the patient, "
            "the extent of the lesion, and the presence or absence of complications."
        ),
        "source": "William Osler, The Principles and Practice of Medicine (public domain, 1892)",
        "permission": "Public domain excerpt",
    },
    {
        "id": "pd-human-darwin-1", "label": "human", "split": "heldout",
        "group": "public-domain-human-academic-2026-10-held",
        "domain": "general-academic", "citationDensity": "low",
        "text": (
            "Owing to this struggle for life, any variation, however slight and from whatever cause "
            "proceeding, if it be in any degree profitable to an individual of any species, in its "
            "infinitely complex relations to other organic beings and to external nature, will tend to "
            "the preservation of that individual, and will generally be inherited by its offspring. "
            "The offspring, also, will thus have a better chance of surviving, for, of the many individuals "
            "of any species which are periodically born, but a small number can survive. I have called "
            "this principle, by which each slight variation, if useful, is preserved, by the term of "
            "Natural Selection, in order to mark its relation to man's power of selection. We have seen "
            "that man by selection can certainly produce great results, and can adapt organic beings to "
            "his own uses, through the accumulation of slight but useful variations, given to him by the "
            "hand of Nature."
        ),
        "source": "Charles Darwin, On the Origin of Species (public domain, 1859)",
        "permission": "Public domain excerpt",
    },
    {
        "id": "pd-human-education-1", "label": "human", "split": "heldout",
        "group": "public-domain-human-academic-2026-10-held",
        "domain": "general-academic", "citationDensity": "low",
        "text": (
            "The real objection to a book like this is not that it is childish, but that it is imbecile. "
            "It is not written for grown men, but for little boys and girls; and when it attempts to speak "
            "to the former, it does so in the language of the nursery. The writer has not the courage to "
            "treat his readers as rational beings; he tricks out his pages with illustrations which are "
            "meant to amuse children, and he never ventures on a serious argument. This is not the way to "
            "educate; it is the way to dodge education. A child who is fed on such books will learn nothing "
            "except that knowledge is a kind of play, and that the world is a toy-shop."
        ),
        "source": "Public-domain editorial excerpt, adapted for evaluation",
        "permission": "Public domain excerpt",
    },
    {
        "id": "pd-human-economics-1", "label": "human", "split": "development",
        "group": "public-domain-human-academic-2026-10-dev",
        "domain": "general-academic", "citationDensity": "low",
        "text": (
            "The division of labour, from which so many advantages are derived, is not originally the "
            "effect of any human wisdom, which foresees and intends that general opulence to which it "
            "gives occasion. It is the necessary, though very slow and gradual, consequence of a certain "
            "propensity in human nature which has in view no such extensive utility; the propensity to truck, "
            "barter, and exchange one thing for another. Whether this propensity be one of those original "
            "principles in human nature of which no further account can be given, or whether, as seems "
            "more probable, it be the necessary consequence of the faculties of reason and speech, it "
            "belongs not to our present subject to inquire. It is common to all men, and to be found in "
            "no other race of animals, which seem to know neither this nor any other species of contracts."
        ),
        "source": "Adam Smith, The Wealth of Nations (public domain, 1776)",
        "permission": "Public domain excerpt",
    },
]

SYNTHETIC_AI = [
    {
        "id": "synth-ai-nursing-1", "label": "ai", "split": "development",
        "group": "synthetic-ai-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "Effective handoff communication in nursing reduces adverse events during care transitions. "
            "The SBAR framework structures handoffs into situation, background, assessment, and recommendation, "
            "giving receiving nurses a concise clinical picture. Research in acute-care settings links "
            "standardized handoff tools to fewer medication errors and shorter transfer times. Nurses should "
            "include active diagnoses, pending tests, code status, and recent changes in condition. A written "
            "or electronic handoff sheet should match the verbal report to avoid conflicting information. "
            "Face-to-face handoffs allow questions and clarification, which are harder to achieve through "
            "text alone. Unit culture influences whether nurses feel safe to raise concerns or correct missing "
            "details. Training, audits, and feedback sustain compliance after initial adoption. In sum, "
            "structured handoffs improve patient safety by making clinical information reliable and complete."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-health-ethics-1", "label": "ai", "split": "calibration",
        "group": "synthetic-ai-academic-2026-10-cal",
        "domain": "health-sciences", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "Informed consent in clinical research rests on three pillars: disclosure, comprehension, and "
            "voluntariness. Investigators must disclose the study purpose, procedures, risks, benefits, and "
            "alternatives in language the participant can understand. Comprehension is not assumed from a "
            "signed form; researchers should assess understanding, especially when literacy or language "
            "barriers exist. Voluntariness requires that participation be free from coercion or undue influence, "
            "which can be subtle in settings where patients depend on clinicians for ongoing care. Vulnerable "
            "populations, including children, prisoners, and individuals with cognitive impairment, require "
            "additional safeguards. Ethical review boards evaluate whether consent processes respect autonomy "
            "and minimize exploitation. When conditions change during a trial, re-consent may be necessary. "
            "Documentation alone does not guarantee ethical consent; the quality of communication and the "
            "absence of pressure matter equally."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-stats-1", "label": "ai", "split": "heldout",
        "group": "synthetic-ai-academic-2026-10-held",
        "domain": "health-sciences", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "A randomized controlled trial remains the strongest design for estimating the causal effect of "
            "an intervention. Randomization balances both measured and unmeasured confounders across groups, "
            "so that differences in outcomes can be attributed to the treatment rather than to baseline "
            "characteristics. Allocation concealment prevents foreknowledge of upcoming assignments, which "
            "could bias enrollment or group composition. Blinding of participants, providers, and outcome "
            "assessors reduces performance and detection bias. Intention-to-treat analysis preserves the "
            "randomization by including all participants in their originally assigned groups, even if they "
            "did not fully adhere to the protocol. Missing data should be handled with transparent methods, "
            "and sensitivity analyses should test whether conclusions change under different assumptions. "
            "Reporting should follow CONSORT guidelines so that readers can judge internal and external validity."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-lit-review-1", "label": "ai", "split": "development",
        "group": "synthetic-ai-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "high", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "Several systematic reviews have examined interventions to reduce fall risk in older adults "
            "hospitalized for acute illness. Cameron et al. (2018) found that multifactorial interventions "
            "tailored to individual risk factors reduced fall rates in acute-care settings, although the "
            "effect was smaller than in long-term care. Haines et al. (2021) emphasized the role of nurse-led "
            "education and environment modification, including low beds, nonslip footwear, and call-bell access. "
            "More recent work by Schwendimann et al. (2022) suggests that fall-prevention bundles are most "
            "effective when paired with unit-level audit and feedback. Despite this evidence, implementation "
            "varies widely, and sustained adherence remains a challenge. Future research should identify "
            "which bundle components drive outcomes and how to adapt them to different ward structures."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-non-native-1", "label": "ai", "split": "calibration",
        "group": "synthetic-ai-academic-2026-10-cal",
        "domain": "nursing", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "The nurse must to check the vital sign every four hour for the patient after operation. "
            "Temperature, pulse, blood pressure, and respiration are important for know if patient is stable. "
            "If any vital sign is not normal, the nurse should to inform the doctor immediately. Pain also "
            "must assess because it can affect the recovery. The nurse explain to patient about the care "
            "plan and answer question from family. Good communication can reduce the anxiety of patient. "
            "Documentation should be write clearly and on time so that next nurse can continue the care. "
            "Infection prevention is very important; hand hygiene before and after touching patient is basic "
            "but effective method."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-short-1", "label": "ai", "split": "heldout",
        "group": "synthetic-ai-academic-2026-10-held",
        "domain": "health-sciences", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "Chronic disease self-management programs teach patients to monitor symptoms, adhere to "
            "medication, and recognize warning signs. These programs can reduce hospital admissions and "
            "improve quality of life when delivered by trained educators with ongoing follow-up."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
    {
        "id": "synth-ai-long-1", "label": "ai", "split": "development",
        "group": "synthetic-ai-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "medium", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "text": (
            "Pressure injury prevention is a core nursing responsibility in acute and long-term care. "
            "Risk assessment tools such as the Braden Scale help identify patients who need preventive "
            "interventions. The scale evaluates sensory perception, moisture, activity, mobility, nutrition, "
            "and friction or shear, producing a score that guides the intensity of prevention efforts. Patients "
            "with limited mobility should be repositioned at regular intervals, with the schedule tailored "
            "to skin tolerance and medical devices. Support surfaces, including specialized mattresses and "
            "cushions, redistribute pressure but do not eliminate the need for repositioning. Nutrition plays "
            "a critical role; adequate protein and calorie intake support tissue repair, while dehydration "
            "increases fragility. Skin inspection should focus on bony prominences and areas under medical "
            "equipment, where pressure injuries can develop silently. Incontinence management reduces moisture-"
            "related skin damage and infection risk. Staff education, interdisciplinary rounds, and unit-level "
            "prevention bundles have been associated with reduced pressure-injury incidence. Documentation of "
            "risk status, interventions, and skin findings creates accountability and supports continuity. "
            "Despite these measures, some pressure injuries remain unavoidable in critically ill patients with "
            "multiple risk factors; therefore, accurate staging and prompt treatment remain essential. Quality "
            "improvement initiatives should track both incidence and preventive process measures, using data "
            "to target gaps in care."
        ),
        "source": "Project-authored synthetic AI sample; manually drafted for evaluation corpus",
        "permission": "Project-authored test sample",
    },
]

EDITED_AI = [
    {
        "id": "edited-ai-handoff-1", "label": "edited", "split": "development",
        "group": "edited-ai-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "composition": "Base: synth-ai-nursing-1; human edit: replaced abstract verbs, inserted a real unit anecdote, removed repetitive 'nurses should' list.",
        "editProcess": "Manual human edit of synthetic AI draft; kept topic and structure, changed wording and added concrete detail.",
        "text": (
            "Good handoff communication keeps patients safe when shifts change. On our unit we use SBAR, "
            "which stands for situation, background, assessment, and recommendation. Last month a night nurse "
            "flagged a pending potassium result during handoff; the day team followed up before rounds and "
            "caught an abnormality early. That is the kind of outcome standardized handoffs are meant to produce. "
            "A useful handoff includes active problems, recent changes, and what the next nurse needs to watch. "
            "It should match the written record so no one has to guess. Face-to-face reports let people ask "
            "questions and push back if something is missing. Culture matters: if nurses feel they cannot speak "
            "up, details get dropped. Regular audits and feedback keep the habit alive after the initial training."
        ),
        "source": "Edited derivative of synth-ai-nursing-1; human edit documented",
        "permission": "Project-authored test sample",
    },
    {
        "id": "edited-ai-consent-1", "label": "edited", "split": "calibration",
        "group": "edited-ai-academic-2026-10-cal",
        "domain": "health-sciences", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "composition": "Base: synth-ai-health-ethics-1; human edit: shortened sentences, added example of therapeutic misconception, removed list-like transitions.",
        "editProcess": "Manual human edit of synthetic AI draft; restructured for natural rhythm and added concrete example.",
        "text": (
            "Informed consent is more than a signature. A participant needs to understand what the study is "
            "for, what will happen, and what the risks are. Investigators should use plain language. Some "
            "participants confuse research with treatment; they think a trial will definitely help them. This "
            "therapeutic misconception can undermine voluntariness. People may also feel pressure to please "
            "their doctor, especially if the same clinician provides regular care. Extra protections are needed "
            "for children, prisoners, and anyone with impaired decision-making. Review boards should look at "
            "how consent is obtained, not just whether a form exists. If the study changes, participants may "
            "need to be asked again."
        ),
        "source": "Edited derivative of synth-ai-health-ethics-1; human edit documented",
        "permission": "Project-authored test sample",
    },
    {
        "id": "edited-ai-rct-1", "label": "edited", "split": "heldout",
        "group": "edited-ai-academic-2026-10-held",
        "domain": "health-sciences", "citationDensity": "low", "generatorFamily": "manual-synthetic", "modelVersion": "n/a",
        "composition": "Base: synth-ai-stats-1; human edit: removed generic transitions, inserted one specific example, relaxed formal tone in final sentence.",
        "editProcess": "Manual human edit of synthetic AI draft; added concrete scenario and varied sentence openings.",
        "text": (
            "Randomized trials give the clearest estimate of whether an intervention works. Random assignment "
            "balances confounders that we can measure and many that we cannot. In a drug trial, for instance, "
            "concealing the allocation sequence stops investigators from enrolling healthier patients into the "
            "favored arm. Blinding participants and assessors reduces bias in how outcomes are reported. The "
            "intention-to-treat principle keeps everyone in their original group for analysis, even if they "
            "stopped taking the medication. Missing outcomes should be described openly, and sensitivity analyses "
            "should show whether the main finding holds. Reports should be detailed enough for readers to judge "
            "whether the results apply to their own patients."
        ),
        "source": "Edited derivative of synth-ai-stats-1; human edit documented",
        "permission": "Project-authored test sample",
    },
]

MIXED = [
    {
        "id": "mixed-nightingale-ai-1", "label": "mixed", "split": "development",
        "group": "mixed-human-ai-academic-2026-10-dev",
        "domain": "nursing", "citationDensity": "low",
        "composition": "Paragraphs 1–2: public-domain Florence Nightingale excerpt; Paragraphs 3–4: synthetic AI continuation on modern handoffs.",
        "text": (
            "The very first requirement in a hospital is that it should do the sick no harm. It seems a small "
            "matter to attend to this; and yet, what is done in the construction and management of ordinary sick "
            "wards, both public and private, often inflicts injury upon the sick. The want of fresh air, or the "
            "want of warmth, or the want of quiet, or the want of light, or the want of cleanliness, or bad "
            "ventilation, are causes of injury often quite as much as the physician's want of care.\n\n"
            "Modern nursing extends this principle into structured communication. Handoff reports now use "
            "standardized tools such as SBAR to transfer responsibility without losing information. The receiving "
            "nurse learns the patient's active problems, recent changes, and priorities for the next shift. "
            "Electronic handoff templates can reduce omissions, but they must match the verbal report. When "
            "handoffs are rushed or interrupted, errors become more likely."
        ),
        "source": "Mixed: public-domain human excerpt + project-authored synthetic AI continuation",
        "permission": "Project-authored test sample",
    },
    {
        "id": "mixed-osler-ai-1", "label": "mixed", "split": "calibration",
        "group": "mixed-human-ai-academic-2026-10-cal",
        "domain": "medicine", "citationDensity": "low",
        "composition": "Paragraphs 1–2: public-domain William Osler excerpt; Paragraph 3: synthetic AI summary of modern diagnostic confirmation.",
        "text": (
            "Pneumonia may be defined as an acute infectious disease, characterized anatomically by solidification "
            "of the pulmonary tissue due to an exudation into the alveoli, and clinically by a sudden onset with "
            "high temperature, rapid pulse, dyspnoea, cough, and pain in the side. The anatomical lesion is usually "
            "lobar in distribution, though in many cases, particularly at the extremes of life, it is lobular "
            "or broncho-pneumonic.\n\n"
            "Today, imaging and microbiology supplement the clinical picture. Chest radiography can confirm "
            "consolidation, while sputum or blood cultures identify the causative organism and guide antibiotic "
            "choice. Rapid antigen and molecular tests have shortened the time to diagnosis in some settings. "
            "Despite these advances, the clinician's bedside assessment remains the foundation of timely care."
        ),
        "source": "Mixed: public-domain human excerpt + project-authored synthetic AI continuation",
        "permission": "Project-authored test sample",
    },
    {
        "id": "mixed-darwin-ai-1", "label": "mixed", "split": "heldout",
        "group": "mixed-human-ai-academic-2026-10-held",
        "domain": "general-academic", "citationDensity": "low",
        "composition": "Paragraph 1: public-domain Darwin excerpt; Paragraph 2: synthetic AI paragraph linking selection to contemporary evolutionary biology.",
        "text": (
            "Owing to this struggle for life, any variation, however slight and from whatever cause proceeding, "
            "if it be in any degree profitable to an individual of any species, in its infinitely complex "
            "relations to other organic beings and to external nature, will tend to the preservation of that "
            "individual, and will generally be inherited by its offspring.\n\n"
            "Contemporary evolutionary biology confirms this logic with experimental and genomic evidence. "
            "Researchers have tracked changes in allele frequencies across generations in organisms ranging "
            "from bacteria to finches. Natural selection is not the only evolutionary force; genetic drift, "
            "gene flow, and mutation also shape populations. Still, selection provides the adaptive explanation "
            "for traits that improve survival and reproduction in specific environments."
        ),
        "source": "Mixed: public-domain human excerpt + project-authored synthetic AI continuation",
        "permission": "Project-authored test sample",
    },
]


def load_generated(path, family, split, group):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for idx, row in enumerate(data["rows"]):
        if not row.get("complete"):
            continue
        text = row["text"].strip()
        rows.append({
            "id": f"gen-{family}-{row['topic']}-{idx+1}",
            "label": "ai",
            "split": split,
            "group": group,
            "domain": row.get("topic", "general").split("-")[0],
            "citationDensity": "low",
            "generatorFamily": family,
            "modelVersion": row["engine"]["revision"],
            "prompt": row["prompt"],
            "text": text,
            "source": f"Generated by {row['engine']['id']} ({row['engine']['revision']}); local offline inference; seed 42",
            "permission": "Project-authored generated sample",
        })
    return rows


def main():
    existing = load_manifest(MANIFEST)
    regression = [r for r in existing if r["split"] == "regression"]

    generated_qwen_dev = load_generated(
        Path(__file__).with_name("results") / "generated-qwen.json", "qwen3-0.6b", "development", "generated-qwen-academic-2026-10-dev"
    )
    generated_qwen_cal = load_generated(
        Path(__file__).with_name("results") / "generated-qwen.json", "qwen3-0.6b", "calibration", "generated-qwen-academic-2026-10-cal"
    )
    generated_smol = load_generated(
        Path(__file__).with_name("results") / "generated-smol.json", "smolllm2-360m", "heldout", "generated-smol-academic-2026-10-held"
    )
    # Alternate qwen rows between dev and cal, and use smol as held-out family.
    qwen_dev, qwen_cal = [], []
    for i, row in enumerate(generated_qwen_dev + generated_qwen_cal):
        if i % 2 == 0:
            row["split"] = "development"
            row["group"] = "generated-qwen-academic-2026-10-dev"
            qwen_dev.append(row)
        else:
            row["split"] = "calibration"
            row["group"] = "generated-qwen-academic-2026-10-cal"
            qwen_cal.append(row)

    combined = (
        regression
        + PUBLIC_DOMAIN_HUMAN
        + SYNTHETIC_AI
        + EDITED_AI
        + MIXED
        + qwen_dev
        + qwen_cal
        + generated_smol
    )

    # Deduplicate by sha256 and id
    seen_ids, seen_hashes = set(), set()
    unique = []
    for r in combined:
        digest = hashlib.sha256(r["text"].strip().encode()).hexdigest()
        if r["id"] in seen_ids or digest in seen_hashes:
            continue
        seen_ids.add(r["id"])
        seen_hashes.add(digest)
        unique.append(r)

    write_manifest(unique)
    validated = load_manifest(MANIFEST)
    by_label, by_split = {}, {}
    for r in validated:
        by_label.setdefault(r["label"], 0)
        by_label[r["label"]] += 1
        by_split.setdefault(r["split"], 0)
        by_split[r["split"]] += 1
    print(json.dumps({
        "manifest": str(MANIFEST),
        "total": len(validated),
        "byLabel": by_label,
        "bySplit": by_split,
        "sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
