# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9bc9ae6eca1afa6b83b6e75670188dd332e3c4f110971932dc47f6d6a315c1b4`
- fixture hash: `sha256-802793623b4c122b10687187b8ed29a08f9e42bf4ed06ad0911f576e3bb3e669`
- score hash: `sha256-9d4f31913ef00f0eb1c43bcc5850aa10ebb0b9d35451875ffce0557daf0c495c`
- bundle hash: `sha256-35f939b505efc1a84af638edab091d86670a54e1fbf98449a4ba509f294997db`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4c6cbf75055b2f2066145c893826e751aed7af61508ee55203c7ec8985a9cd38 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7e9c731d7c452ebc71d70f08e0af927f321858784c90e4a54cacd6203bbda460 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c1255fbfcf8891df92581f6f2510e847820073a7684c04880ab45c181de36c1 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-749a5932a52df90237c65441e2ede2e0520d877057c0f82cb7dd8aadbfc3f3f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-acbfb32b | sha256-43ccbf17b07b7b6b00dbf5781c5e5c6389d4d17b0248646469a247aab5c450b6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-acbfb32b | sha256-ee4e1572d9628095def827250a7c4eaa84fa75253b37ff4cb7120196fc6acb1b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-dba2a674 | sha256-e01e0afd89dd9a2687d98e110fb9d7ad6a54e3091a9efe08f831c81b03efa532 |
