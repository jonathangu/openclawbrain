# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-177`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22c0c5cfe30f6528627aae6b3b1ce6c55137840c4388f7d03d5ba0c64043e114`
- fixture hash: `sha256-883333e2877ee56be18afd0bdb26f3a044eab5df448e40bf59cfd947e2e070a7`
- score hash: `sha256-7064d0835649437a3cd4e9996456ebde192bc2dfce858ea1e83fbd4a24414ed3`
- bundle hash: `sha256-76f2a36c083d1ee0e99e7935f4b75374382721d6f07a15bdd6382681de95b441`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bfacda8f5501f5e4f01bbebcdaf7a5c0e18d211755bb5803d41f576de0d46bba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64f60c63236b59c114d3b19ed956a5bc274c58dfe79ce9b78d6dbd7b94316e22 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8f18feb3c780753f78a5d30ad1d7a6634cfe495af0f5a6eee4d73d4569147929 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-907698adba53303be1aed10506b04bdb11abdbfeb69d1cb8f5475d2b7732d9ea |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dd3b22f2 | sha256-07ce4f2b33a4db130ba83bac0e79025599ce116708d8b9b650067f24e7a3aedd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dd3b22f2 | sha256-eb183329d1a4a032e0f5a2a11e902dcb7babb92f67dbf878518eb5446b4b8077 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cf1163a3 | sha256-fe2d44e1ee64fe3256696950f0abedad81638160b14afb24651cf465c6c3c567 |
