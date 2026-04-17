# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bfcdb554e6c3bfe187f4c905f92a9b282d7821367cef535897c2815e123fe75d`
- fixture hash: `sha256-3907274214cdd60210f9dcb9d9b0e865d090d5365a59db918b98e4ad4849f4e5`
- score hash: `sha256-cd6582a111fcb3eb75cf2223394143d2e4f6dfd073578543bf85315eb5fd70f0`
- bundle hash: `sha256-c396e53d89a36ed3ab17ea162a6eb4cd5c1f8cd30a3dd626cbad33729a3c3c95`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18ae191771eabc01fba0eef9c0e7f277194aa1ae188e2e94481f667ee00cc41c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-08ed08da4663ce2d4cf403f508b11ca162250e5e9838e80b3a67234dc9f1cf7d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf6bcd9bb6638c09239828e2b91faef4048f709a746a8f5f9f18b9317b478b68 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-1a8ad12d2948650913073a3a337fec0386229196147b53a5d072e7da254e1d53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-65e49deb | sha256-449faf38debbca4fb52da2f35617250968074141a87cb9229e8b4f0cb4548ae7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-65e49deb | sha256-98ecfc7c131f34f40ffa377033524da703f0cf91772c9f18619510fe6f63bdc7 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-33cbb574 | sha256-15554cab640a6b1f13a0646a61761d435dddd5f0d93bbecb649f80c86457d020 |
