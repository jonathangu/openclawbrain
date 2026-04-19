# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e204eec37724aacda490110857d04f610417247b024d310ad6302080aea397ff`
- fixture hash: `sha256-6f87b65c79b51cbc945548d32dcf271be29d7e50d05ff0c454ef7979a8b75cf2`
- score hash: `sha256-e1cb27738724e056f69dc4e3289908815eae6a5e91afc895675a5e3eea0dbab0`
- bundle hash: `sha256-ace730faea6e0ffc2378143a243fd22b7c7c47029f964f4a7441b3f7ce298032`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c76acc1a03aa06534fe63599a6439df3e7c5ba77b6ef580f7d358b50380fb3e8 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-796bf6fc122e91376fe322bcfb9dd4e6f38baf1a9ec9baa40c48f6c4bf5e8579 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-20076e3bfae8192a44fe8641d084aa8681798386ad86338114d75b9453564076 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8c00541d4f9a0464ee38b862a04727f94e97da1fa6c1349b972e1f69655db9b5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-a4e80e43 | sha256-9a748addd659a57afd2b43d4e9e37cbf2fc0343a17bb536ba7519c0d3ed51677 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-a4e80e43 | sha256-d7b790f7c5a62d64750355836f74a1a44b403ba39d0a5aea8ae4424c69ced000 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a4e80e43 | sha256-dfcaa2708d25dbb7f00f0cab969108eab7643d856ec6fee42a1c86c7b93d3178 |
