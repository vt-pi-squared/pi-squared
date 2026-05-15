# Reasoning Trace Generation Prompt

You are given a question, a set of context documents/chunks, and the correct answer.
Your task is to generate a detailed, step-by-step reasoning trace that explains how to arrive at the answer by analyzing the context.

## Input Format

### **Context:**
{context}

### **Question:**
{question}

### **Correct Answer:**
{answer}

## Output Format Requirements

Generate a natural-language reasoning trace following these structural patterns:

### 1. Step Numbering
- Use `* Step X:` format for major steps
- Steps should progress logically from understanding the problem to arriving at the answer

### 2. Planning Steps (Initial Steps)
- Start by analyzing what the question is asking
- Identify what information needs to be extracted from the context
- Determine the search strategy (e.g., "find all candidates for each criterion")
- Example: "The question asks about X that fits Y criteria, so we need to find all candidates..."

### 3. Extraction & Navigation Steps
- Systematically review relevant documents/chunks
- Document your search process (e.g., "let's search for keyword X: found in documents A, B, C")
- For each document examined, state whether it contains relevant information
- Use phrases like:
  - "Let's find..."
  - "Document X says..."
  - "It fits/does not fit because..."
  - "Let's check..."
  - "Move to document..."

### 4. Iterative Filtering
- For multi-criteria questions, evaluate candidates against each criterion
- Narrow down the candidate list as you verify criteria
- Explicitly state when a candidate is eliminated and why
- State when a candidate passes a criterion

### 5. Evidence Citation
- Always cite specific documents/chunks when stating facts
- Quote relevant passages when they directly support conclusions
- Use formats like:
  - "Document X states: '...'"
  - "According to document X,..."

### 6. Self-Verification
- Include steps that cross-check findings
- Verify consistency across multiple sources when available
- Address potential contradictions explicitly

### 7. Reasoning Steps
- Perform virtual operations (filtering, sorting, comparing) explicitly
- When calculating, show the calculation
- When comparing, state the comparison clearly
- Build confidence through multiple verified facts before final answer

### 8. Conclusion
- The final step should clearly state the answer
- Briefly summarize the key evidence that leads to it

## Style Guidelines

1. **Natural and exploratory**: Write as if discovering the answer in real-time
2. **Self-contained**: Each step should be understandable without re-reading previous steps
3. **Critical evaluation**: Don't accept information at face value; evaluate relevance
4. **Document-focused**: Ground all conclusions in the provided context
5. **Conversational but precise and concise**: Use natural language while maintaining factual accuracy and efficiency

## Example Structure

```
* Step 1: [Understanding the question and planning strategy]
* Step 2: [Search for candidates matching criterion A - examining documents]
* Step 3: [Search for candidates matching criterion B - narrowing down]
* Step 4: [Cross-referencing and verification]
* Step 5: [Conclusion with answer]
```

## Important Notes

- The reasoning trace should be answer-agnostic in style (don't reveal you know the answer upfront)
- Show the discovery process as it would happen when solving the problem
- Include dead-ends and corrections when relevant (e.g., "Wait, this document is from 2018, so it may be outdated")
- If multiple interpretations exist, explore them before concluding

Now generate the reasoning trace for the given input.
