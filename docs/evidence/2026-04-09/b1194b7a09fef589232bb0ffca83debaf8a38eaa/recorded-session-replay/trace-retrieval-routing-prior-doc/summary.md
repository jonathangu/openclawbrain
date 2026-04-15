# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-routing-prior-doc`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f541ef591a3fdeca44ec33d94d63586bddb7736c330abf7d4032054bf4d0e239`
- fixture hash: `sha256-10d2f6679e370351117cc6864a5d47c336bab40268450d59ff126a9b849b1bc2`
- score hash: `sha256-50e7e482137879f0d8929a049d60b77407662a5852ea73b52a4678fd5867577e`
- bundle hash: `sha256-a79d1d3ae3c7c6c5060945970c36b324c019cf5bdebc8bb35c73691b34cbcda4`

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
- phrase hits: 9/12
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
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-05be127f6e13c4988ff9e6455aa8abdfef90f76ab4fa10fd101198d4fa2025d5 |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-57567677e4dbca39810dce883b5a749588e6da3ff38942dfab179d9a2bb81724 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-7494442e5c60a079c05b3ef83c1317fcba5df06f1c8853ba6b960888ed34e11d |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-5f1d80921fbf19205dee3e9f0e46346e6f19f3db0807966a3a6e572d8a46284a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | routing-docs-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | routing-docs-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | routing-docs-turn-1 | 100 | yes | 1/1 | no | no | pack-aee3068d | sha256-8ab82aab9fa02002b1b7a717dc1762f264f8eb1a70b67cc102ee61af80e48bdd |
| vector_only | routing-docs-turn-2 | 100 | yes | 2/2 | no | no | pack-aee3068d | sha256-8ab82aab9fa02002b1b7a717dc1762f264f8eb1a70b67cc102ee61af80e48bdd |
| graph_prior_only | routing-docs-turn-1 | 100 | yes | 1/1 | no | no | pack-aee3068d | sha256-8ab82aab9fa02002b1b7a717dc1762f264f8eb1a70b67cc102ee61af80e48bdd |
| graph_prior_only | routing-docs-turn-2 | 100 | yes | 2/2 | no | no | pack-aee3068d | sha256-8ab82aab9fa02002b1b7a717dc1762f264f8eb1a70b67cc102ee61af80e48bdd |
| learned_route | routing-docs-turn-1 | 100 | yes | 1/1 | no | yes | pack-aee3068d | sha256-8ab82aab9fa02002b1b7a717dc1762f264f8eb1a70b67cc102ee61af80e48bdd |
| learned_route | routing-docs-turn-2 | 100 | yes | 2/2 | yes | no | pack-c38d4ece | sha256-c8d18da340c246a871da0bb86f539d361e1cb24fdbba31865edf9e6a1b593629 |
