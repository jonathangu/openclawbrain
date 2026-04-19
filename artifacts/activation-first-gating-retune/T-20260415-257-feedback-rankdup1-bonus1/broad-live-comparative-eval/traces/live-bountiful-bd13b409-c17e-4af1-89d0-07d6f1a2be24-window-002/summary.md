# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-bd13b409-c17e-4af1-89d0-07d6f1a2be24-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-84f2bc85f266bfd29b006c2dc6ef93b9e165e29f55fc2011158e761c60436082`
- fixture hash: `sha256-810d1c824da649ceb29c027b026aba0238fc9fe617410c118b1a8cf769536f9f`
- score hash: `sha256-d9d0298131534afaab7a189450da2aaac99b78ee1657de92d07165e4129ebb57`
- bundle hash: `sha256-dfba1aef6da5cc20a35efbb3f0e4861c588a78f5d90841ad18e5d70229b0de9a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1065d3831f028776c526f039887471c1da0a0d15099e1e578d4ef85bdda6d0d2 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-8e93e2d931d7e0115fcfe3beb0e64ea071f19a3dee6ea7885b3d974d6dfd6835 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-dce5375e87443dbbe1e5d84b98f099b7be8510f698b8a968d21cdbbe17c5d4fc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c34620bdd8e30085fa9fdf5df7746d5cec3207e695f9adb5223b8e515ece25d6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-153f294a | sha256-a8e355973eaa90e5a97457a037e0af42e479c55aa076e8e268323a5948be6702 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-153f294a | sha256-8db4f0952072c6cb0a1f74925c2ad33b62137cf06a3be28644631ed1bec3f205 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-153f294a | sha256-a8e355973eaa90e5a97457a037e0af42e479c55aa076e8e268323a5948be6702 |
