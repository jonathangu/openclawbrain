# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1078b1a70bcd22daa0ead376beedaa52bfe2cf8765ec6a491cb29b47f4429da`
- fixture hash: `sha256-48416b4518f830c212c5a38183605df066ce4a1235bd3582b824c27bcab21c53`
- score hash: `sha256-49013d857e5e5e1c55cce3e1831b24cecd2bc868f99a3717ce82b8c03c27abfc`
- bundle hash: `sha256-b42fe61cabeb4f93dd9d8928bb6d7e1e2307dacc0a514bbee64a63c0341d60f5`

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
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57e6cc1ff0fcf88903029010179cd9e85affa629951b704a6bd53f2a38e4810e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dda22d68a4104ba47d0fe5617001bb6390d1f5781434a35f869aa0c692e85778 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d0a012afce89b46a3a1bbe952ac25ecedaaebadaefca5d0849ceb640eb178442 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d03708a9e7b81c4abbba3ad0640e6bc9cedf7952d756d22b0bf01088eff1cd0c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2964bd81 | sha256-07ccbcbf9b2ecb6de756ce6c7fde318b6c6cbe6b266a77c25aa6dab86661992f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2964bd81 | sha256-07ccbcbf9b2ecb6de756ce6c7fde318b6c6cbe6b266a77c25aa6dab86661992f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-561f576a | sha256-cd57a189527c2f247e0e82de699bb729b7389a7652ce6f84fce3547cff6918c5 |
