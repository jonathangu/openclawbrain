# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1fd020946f43a8035e7b5bddd63932b365f2da8b3586893d3d0f370ca217a92d`
- fixture hash: `sha256-9e494f0e3b9885b669e057408d4e8e8d1d86d287d75fdec54a2b7380f98f07fa`
- score hash: `sha256-1545f5ad31ec4058b700dc613290582445474acffec406c2589462ecf56866d3`
- bundle hash: `sha256-2a0ab062d186d3525b30bcadfa58b85af6fdaf01e5e9a1a08a393e351b901778`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f82956037e9003d716b0eb45a53efb59bb4a7228c918209cdbf92b9a2ea73fc |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8e62ef0d6b538b451eb99c1272658467b6b0c28f67572ac6a5ec3b60721cb948 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-2ddcb132921587597b6575a3a3e63756d4f597fba3ba4dd9c2674931f99f5689 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0fba91a6f2fa08fab413b367c493b20f45a647a22021be86a8d7602562a0757f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5c27ba27 | sha256-3f998662502a3dcb77203ac6655d73df74aeeafcaffdd5c70864f5781a42d181 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-5c27ba27 | sha256-0f7e9f8995780dc9c00af8fcce199d491f984a2ede7bea609c111bd90754cb12 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5c27ba27 | sha256-3f998662502a3dcb77203ac6655d73df74aeeafcaffdd5c70864f5781a42d181 |
