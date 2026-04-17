# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c4b362666455e64ccec1b7026696e6d7ee86b07af9d91203554a5f880643a7f`
- fixture hash: `sha256-ed322207ac696cb8afb94d5f75c53ba4423b96ed55d3a35abcef96ba37d6147e`
- score hash: `sha256-968295c03bb37f423ec97312c984b886bdf229b7818dab434ec197b2cd94bf7e`
- bundle hash: `sha256-2019c74576b893d7df94a894ce4f626012a84dc9c8a21272fdf9757ae2c3768c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-659f0a773cf1e71603ffb20a5a28aebea6d5db6139dfdf0255a605e7868cd22c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b3ccea6b49dace90bc8cd6a59fdda73b21dc5bbc07439171aa3842f70bde7510 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-466e4384f5f1a22c72fb42b3fd706feddac3125c762a0a5bdca3f05921fe0eb8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-48473ce6e8c6c831dcb30127d5758c47b51893620d8fe9b84c959cca22d1ed70 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa80623d | sha256-ca97b94caee90a15010bc44fa162f53e10a09bac05593140b315cc6ad4f48470 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa80623d | sha256-fa7bfd4fee91f599587bfd2f85bc48af51c0904106b7d080d09e15f63401d4dd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-78ce2af4 | sha256-e2d1a5313b5bfc2a580d1595afc345ffa7613291f970bfadc3b25802acf398c8 |
