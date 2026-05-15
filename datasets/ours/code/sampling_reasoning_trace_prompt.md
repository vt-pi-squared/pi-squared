You are given a question and its context, which can be a table, a long span of text, or multiple retrieved documents. No ground-truth answer is provided.
Your task is to analyze the context, infer a best-supported answer, and generate a detailed, step-by-step reasoning trace that explains how you arrived at that answer using the evidence in the context.

The generated reasoning trace should follow these structural patterns:

1. Step Numbering
- Use `* Step X:` for each major step.
- Steps should progress logically from understanding the question to producing an answer and justification.

2. Planning Steps (Initial Steps)
- Start by restating what the question asks and what information is needed from the context.
- Describe the search strategy (e.g., keywords, criteria, document order).
- Example phrasing: "The question asks about X, so I will search for Y and Z keywords across the documents."

3. Extraction & Navigation Steps
- Systematically examine relevant documents/chunks.
- For each document reviewed, state whether it contains relevant information and quote or paraphrase the supporting passage.
- Use phrases like:
  - "Let's find..."
  - "Document A states: '...'"
  - "It fits/does not fit because..."
  - "Move to document..."

4. Iterative Filtering
- If multiple candidate answers are possible, list them and evaluate each against the evidence.
- Explicitly eliminate candidates that lack supporting evidence and state why.
- Mark candidates that pass each criterion.

5. Evidence Citation
- Always cite specific documents/chunks when asserting facts.
- Quote relevant passages when they directly support a claim, using the format: "Document X states: '...'"
- If documents conflict, present the conflicting passages and evaluate their reliability.

6. Self-Verification
- Cross-check findings across multiple documents where possible.
- Note contradictions and state how you resolve them (e.g., prefer more direct or recent evidence).
- If evidence is insufficient to decide, say so and list what additional information would resolve the uncertainty.

7. Reasoning Steps
- Perform explicit operations (filtering, comparing, counting) and show them in the trace.
- When performing calculations, show the arithmetic.
- When comparing candidates, state the comparison clearly.

8. Conclusion & Answer(s)
- End with a clear answer section:
  - If you can confidently infer a single best answer: state it and summarize the key evidence supporting it.
  - If multiple plausible answers remain: list the candidate answers, state a confidence level for each (high/medium/low) and the primary supporting evidence.
  - If the context is insufficient: state that no definitive answer can be reached, and list the missing evidence needed.
- Include a brief summary of the most important citations that led to the conclusion.

Style Guidelines
- Natural and exploratory: write as working on the question to find the answer.
- Self-contained: each step should be understandable without re-reading previous steps.
- Critical evaluation: do not accept statements at face value; evaluate relevance and strength.
- Document-focused: ground all conclusions in the provided context.
- Concise: be precise and avoid unnecessary verbosity.

Example Structure
```
* Step 1: [Understanding the question and planning strategy]
* Step 2: [Search for candidates matching criterion A - examine documents]
* Step 3: [Narrow candidates using criterion B - eliminate some]
* Step 4: [Cross-reference and verify evidence]
* Step 5: [Generate hypotheses and assign confidence]
* Step 6: [Conclusion with final answer(s) and supporting citations]
```

Important Notes
- Do not assume facts not present in the context; if you need to infer, label it explicitly as an inference and explain its basis.
- When uncertain, present alternative interpretations and the evidence required to choose between them.
- The trace should show the discovery process; include dead-ends and corrections if they materially affected the outcome.
- Keep all steps grounded in the provided `context`.