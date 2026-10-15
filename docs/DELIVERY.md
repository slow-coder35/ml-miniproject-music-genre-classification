# Submission checklist

1. Fill `team.json` with the real names, student IDs, section, and repository URL. Rebuild documents after changes.
2. Review `results/results.csv`, the prediction records, and the model limitations. Results come from local runs, not the reference report.
3. Submit the two-page PDF unless faculty specifies one page. A one-page alternative is provided because the guidelines conflict.
4. Review the eight-slide deck and rehearse the live app demonstration.
5. Create a **private** GitHub repository and push the source. The current saved GitHub authentication must be renewed with `gh auth login`.
6. Attach the model bundle as a private release asset. Keep the demo model, `demo_model.json`, and matching feature/label metadata together.
7. Add the assigned faculty and TAs as collaborators and confirm that access was accepted.
8. Submit the repository URL, required PDF, and presentation through the faculty-designated portal. The supplied deadline has passed, so use the revised deadline confirmed by faculty.
9. Keep a local copy of the model, documents, and demo recording for the review.

## Model bundle restoration

Extract the supplied submission ZIP to a directory of your choice. It preserves `results/models/` and `results/demo_model.json`. Install dependencies and launch `scripts/run.sh app`. Dataset download and training are not necessary for saved-model prediction.

## Individual contributions

Record only work actually completed by each person in `docs/CONTRIBUTIONS.md`. Both people should understand data leakage, mel features, model comparisons, evaluation, and the inference code. Do not represent an AI-generated commit as a student's individual contribution.

## Preview limitation

The browser permission check denied opening the local app during this implementation session. Streamlit's local application tests exercise loading, uploads, prediction display and error handling, but they do not replace visual browser review. Before the live review, inspect the app yourself and record a short demonstration if required. The slides and PDFs are rendered and visually inspected separately.
