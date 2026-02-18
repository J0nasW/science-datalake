The outline currently has 4 figures, but they're somewhat conventional. Here's what I think would actually make an impact:

  Figures that showcase the integration (the whole point of the paper):

  1. Cross-source citation disagreement heatmap — Not just "r=0.76" in a table, but a 2D density plot (S2AG vs OA citation counts, log-log) with
  color showing where the sources diverge. The V4 finding that one paper has 257K citations in S2AG and 0 in OA is jaw-dropping — that should be
  visible. A Bland-Altman variant would show systematic bias by magnitude.
  2. UpSet plot of source overlap — With 6 boolean flags and 34 observed combinations, this is information-dense and visually novel for
  bibliometrics (common in genomics but rarely seen in scientometrics papers). Much better than a Venn diagram.
  3. Temporal coverage by source — A stacked area or ridgeline plot showing each source's publication-year distribution. This would immediately
  reveal structural differences: OpenAlex goes back centuries, S2AG is CS-heavy and modern, SciSciNet cuts off, RoS has patent-lag. One glance
  tells the whole story.

  Figures that showcase the ontology bridging (our methodological contribution):

  4. Embedding space visualization — UMAP of the BGE-large embeddings showing OpenAlex topics as points colored by domain, with nearest ontology
  terms annotated. This makes the abstract "embedding similarity" tangible — you can see that "Machine Learning" lands near CSO's "machine
  learning" and EDAM's "Machine learning" terms.
  5. Ontology reach comparison — A grouped bar or radar chart: for each OpenAlex domain (Physical Sciences, Life Sciences, Social Sciences, Health
   Sciences), show which ontologies contribute mappings. This reveals that MeSH dominates health, CSO dominates CS, GO covers biology, etc. — the
  multi-ontology approach is justified visually.

  Figures from vignettes (1 highlight each):

  6. Vignette composite — A 2x2 panel with one striking subplot per vignette. Rather than 4 separate figures, a single composite panel saves space
   and shows breadth.

  What I'd cut from the current outline:

  - The architecture diagram — useful but not visually striking; better as a supplementary figure or simplified inline schematic
  - The similarity distribution histogram alone — subsume it into the embedding space visualization

  So the figure lineup would be:

  ┌─────┬────────────────────────────────────────────────────────┬───────────────────────────────────────┐
  │ Fig │                        Content                         │                Novelty                │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 1   │ Cross-source citation Bland-Altman (S2AG vs OA vs SSN) │ Shows why multi-source matters        │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 2   │ UpSet plot of 6-source overlap (34 combinations)       │ Shows what the integration looks like │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 3   │ Temporal coverage ridgeline (8 sources x year)         │ Shows structural bias per source      │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 4   │ UMAP of embedding space (topics + ontology terms)      │ Makes ontology bridging tangible      │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 5   │ Ontology reach by domain (which ontologies cover what) │ Justifies multi-ontology design       │
  ├─────┼────────────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ 6   │ Vignette composite (2x2 panel)                         │ Demonstrates analytical payoff        │
  └─────┴────────────────────────────────────────────────────────┴───────────────────────────────────────┘