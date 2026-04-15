# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adbfb582784ce9c57067bcd682b42040f9ff5a4fc2a41a6b215fa1e5e63926e2`
- fixture hash: `sha256-1b81b9ebc5b6e57a68ac36d63b63963fa7e0e03c9b05269658a97fc89e8025b0`
- score hash: `sha256-15be9448e45861fb53687758a14b94ebc4bf7f353a60b0cb6b05da0a65924464`
- bundle hash: `sha256-a2901f4bd5c6725c0715d6a553e8ea4a8cdc06ba53f84fc5442dc226d0a1bd9f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d918c89a43fc84e7a627af305c1d796a487842c9f1cf040b6474472ff6068ba |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f00723c0d2146e7b2af00ec87a8065010a42c56dd5a7b7f06f9fe8bffd9907b4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aa27b5070eeb2bbfb9312963f30a94389ad21bfad6ea0c38ea7b48563c26e10d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9cef9aa5e50a8e70ce80cc8f2f3fc6f2ab30c148c384980b599aa722e4c83098 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d0a0a6af | sha256-e65c41574149753bf0b152acd531b14507099736c94aed5bf1d2faf55dfbe0b1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d0a0a6af | sha256-264798f413ec01875793b243b0394218227bfa6af4b5082a4567c94c66ebc40a |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d0a0a6af | sha256-e65c41574149753bf0b152acd531b14507099736c94aed5bf1d2faf55dfbe0b1 |
