# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c1b296177f077b6c8091fca65eb450be8d5f631873f87466a2fe9011d8b7c085`
- fixture hash: `sha256-56e21e2f0877b996d5170fefdce01e8f6c2815e782b17ac6f82fa56c1dd0500c`
- score hash: `sha256-ad2f2d3e1df94fb6193d84375079c51c40bf3b0c6ad736d568b66ada788aac3d`
- bundle hash: `sha256-1190bb5afdc9193dfa1742a55a09983372300649702bae80437bab0d4c23cade`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29c5109446f744c540ec8fb2d0eb8a2d5f87ccaaa85851914bbf19fee8f8ade5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-187389b1f448a3adffd968cd3d5e5239ba6670088fc5f3f51b0badc3c9e37aa7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-db02e01961aa220411d31a2a7906dd923e4f16da9d8ecb0a651a3060b167f45f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e5d5b0b86f0e0a51c618d6fecca9b9b6ed1f1d60445a2ef443ec4942c8844467 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ca9a0218 | sha256-afab58a986742a1e5a5d70994087996f33c684bcecd5abfe154dc5f6c261f1ed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ca9a0218 | sha256-874a67f69c8952dbb2786d3e5f7179f498a071b3ea3f5ad6d4074c92f43c710e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9199bf11 | sha256-d98fc0221bd5a071893ac78a08261a8e54231c7f19bbc6c98f1b12bd6d2300bf |
