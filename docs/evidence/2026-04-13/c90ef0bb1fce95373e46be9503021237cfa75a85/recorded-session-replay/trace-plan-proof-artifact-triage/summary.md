# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-proof-artifact-triage`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf8cb60e9ada33f76aa46557e599749091e81d486af58269182793f5706eb5b`
- fixture hash: `sha256-d2de95f47483d717328b536c37387d67416a65c6ef4706bb5e261a3f2a08590e`
- score hash: `sha256-82afa5918c639a74824fea5f8f860e704e13b9499dc3e8a562e5fc54eeb94ee5`
- bundle hash: `sha256-7a2ce481ef191d29e43e1deb6c039e5bec305be9682ea90986b45145a36d8ecd`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 18/24
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/6 | 0 | 0 | 2 | 1 | 0 | sha256-91cb4e3cbbf02d6e28f4e5844a1cdc601a411cf0f2238175efe87f3839231475 |
| vector_only | 2 | 2 | 6/6 | 0 | 0 | 2 | 1 | 0 | sha256-8d3c3764df97b887ae4b330ad71b0f62a49a7ef1cb61d53b40a17695e196108a |
| graph_prior_only | 2 | 2 | 6/6 | 0 | 0 | 2 | 1 | 0 | sha256-7b2944edc1e93fa20523050275db9dd8b1f8b19339eb7b8fb3187c2798f8a72b |
| learned_route | 2 | 2 | 6/6 | 1 | 1 | 2 | 1 | 0 | sha256-69466db008ba721aab4b241a68fb108c6e051b06c69fabbfc8fe382991ed4d98 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | artifact-triage-turn-1 | 0 | no | 0/3 | no | no | none | none |
| no_brain | artifact-triage-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | artifact-triage-turn-1 | 100 | yes | 3/3 | no | no | pack-7343f72e | sha256-027dc2fafd73e994fc97510cbae88818d267832dab89d6ecc120e21d20d85a64 |
| vector_only | artifact-triage-turn-2 | 100 | yes | 3/3 | no | no | pack-7343f72e | sha256-f15ee837aa3687133ca9811e8e0146f3768c3fe9c9c612a429d30a68380ebda7 |
| graph_prior_only | artifact-triage-turn-1 | 100 | yes | 3/3 | no | no | pack-7343f72e | sha256-027dc2fafd73e994fc97510cbae88818d267832dab89d6ecc120e21d20d85a64 |
| graph_prior_only | artifact-triage-turn-2 | 100 | yes | 3/3 | no | no | pack-7343f72e | sha256-f15ee837aa3687133ca9811e8e0146f3768c3fe9c9c612a429d30a68380ebda7 |
| learned_route | artifact-triage-turn-1 | 100 | yes | 3/3 | no | yes | pack-7343f72e | sha256-027dc2fafd73e994fc97510cbae88818d267832dab89d6ecc120e21d20d85a64 |
| learned_route | artifact-triage-turn-2 | 100 | yes | 3/3 | yes | no | pack-9b53bdb3 | sha256-39ee0ff853f1979cd1043de929a1cda0443747ea12c3a486a874229faf307e47 |
