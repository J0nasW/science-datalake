# Ontology Alignment Annotation Guidelines

## Task
For each pair of (OpenAlex topic, ontology term), assign one of three labels:

## Labels

### `correct`
The ontology term is a semantically accurate match for the OpenAlex topic.
This includes:
- Exact or near-exact matches (e.g., "Machine Learning" -> "machine learning")
- Equivalent concepts with different naming conventions (e.g., "Deep Learning" -> "deep neural networks")
- The ontology term covers the same core concept, even if scope differs slightly

### `partial`
There is a meaningful semantic relationship, but the terms are not equivalent:
- Parent-child relationships (e.g., "Convolutional Neural Networks" -> "neural networks")
- Sibling concepts in the same domain (e.g., "Random Forests" -> "decision trees")
- Overlapping but distinct concepts (e.g., "Bioelectronics" -> "Biosensors")

### `incorrect`
The terms are unrelated or the match is spurious:
- No meaningful semantic relationship
- Surface-level string similarity without conceptual overlap
- Homonyms matched incorrectly (e.g., "Mercury" the planet vs "Mercury" the element)

## Guidelines
1. Judge based on semantic meaning, not string similarity
2. Consider the ontology context (e.g., a MeSH term is biomedical, CSO is computer science)
3. When in doubt between `correct` and `partial`, prefer `partial`
4. When in doubt between `partial` and `incorrect`, prefer `partial`
5. Fill in the `label` column in the TSV file with exactly one of: `correct`, `partial`, `incorrect`
