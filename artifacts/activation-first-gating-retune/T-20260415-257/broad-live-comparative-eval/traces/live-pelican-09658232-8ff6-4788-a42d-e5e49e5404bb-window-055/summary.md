# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-055`
- winner mode: `graph_prior_only`
- trace hash: `sha256-069c659e483d79099c9522902169a0e3c008a2a3a1a608f281e5842abe60c793`
- fixture hash: `sha256-2053a334b00cb8986b08e94b050daa206b7253e27c1f42496d3a7ffe4c19e5d6`
- score hash: `sha256-edf5a5525fffa6d9ebdeb984f14ea14b6cc17daa27ba722996aca28374e2005a`
- bundle hash: `sha256-eb7eac135cc382ebbca47390fdd9092219f11560f9b89492bba06c191241eff0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c7d0a9be6adcdff721e255d979e1bead77026cd16f5da5ab306eca424cee158d |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f4aca67c0b511030eb0c1f664b6fd4da8f5eb08b5137da9939a127514835d500 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fd5ab8c984ca2e69196990b255cd432be1b5d76fe5914834c0c3bc1dad6defef |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-9bfe212cdb1e820ac7d61cfbc5825e421b3ddfa7cf0c908be9e5de4199103b90 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-315808e9 | sha256-6e3bd9e97ace0875f936715d92c87b62211fcd570575c52a734cc8685585f28f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-315808e9 | sha256-ba496b1b6896976a16d43a3bad29bfad74655511f4d6c13c7922a4e830bd6ad9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9468a8dc | sha256-9b7ee0a46a5f97f33c3509b576e75b723e64c0b192f5870b3eaed768c6abee82 |
