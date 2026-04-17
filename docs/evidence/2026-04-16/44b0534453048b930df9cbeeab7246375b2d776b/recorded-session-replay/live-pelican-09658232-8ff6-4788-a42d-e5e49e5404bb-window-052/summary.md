# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eb4ce4c16a0b4086f9bf16153627317bccc66c138e3f3eabb740de5aad356d3c`
- fixture hash: `sha256-f8df6b8b0d3896e4d68df7e66273fb59a221dba4842848c8bd3431e1201171eb`
- score hash: `sha256-59779954175b198b320a16b9b09fc3ca7b91a57a076810ee58dd387bfd3f9fc3`
- bundle hash: `sha256-3f810429548367963cb7fd0707b53c0580a182f86ff2525c2fbb18801f76ed5d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31a831566ed58685dea8a0c35a91e51999c06d52779d7820057deddb5dbf99cd |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e4796a37c7bd1cd5fe3e837d7b7ccbeb5ebe479a4c87d2ef16e4ef6b455b5401 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-12e2440381041226b08d64ef36fef06439a11f2a22075c4a90ed76b936d14fdc |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-365df434191b81c930370b4e553bf1116ccf00f22fd17395c226811c587dbfa3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e07e6ffc | sha256-0c9423a641a0e362b5230e5bfe6e4808cc762346e039b40e5bbe20954ae80e19 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e07e6ffc | sha256-6cb5b73debc4848c650da8cb30670987a38be90150eb38b33a540360e61d9c2f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-77a3ada7 | sha256-f58cc6c4c0e9804eaefa70d4ff0e9615623eae124f8ed98f20935b63ecb4a46f |
