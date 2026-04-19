# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ea42e44c1f4e55e54482460d49f94e95de2212323681b79ab4ddfefd7592f32d`
- fixture hash: `sha256-7145b0c661dfe4e1efd2da3fdd776c9b7910c0b1be56f04f06cb4a5ee20cf473`
- score hash: `sha256-81ef3b2908f2e22fef67511e31413dddc809a4902258b7fc589b224de9a413d6`
- bundle hash: `sha256-8acd32709d9b42907e0e4c0c8e9b220c3f4c8d9089bf6816dab22de21981785c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4c76d7fd1d12ceb5bfd4785d2783131a1a6cdf46c225ec13d0667a11e9c25468 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-3b05a79237bd81775413aceefc0ddd9d4ede61fbae56bc26d5d970dd3f6d25d8 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-49352ba1123db598265fd40c5dc8085d75356885e1988c5f19c40e47fcf4368e |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-38f924c68e43c9066e90c8188e93ff4727067e5baa046abd050fa98d2fec8877 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-495164bf | sha256-0a43041cdf5d6bc84f99c2916650e5aef257eb047c2483f94f01f419de131357 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-495164bf | sha256-c105e4799413c3ead0192e420d4b36bc7d3da56bd148372ade4c8d056d580eb5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-495164bf | sha256-0a43041cdf5d6bc84f99c2916650e5aef257eb047c2483f94f01f419de131357 |
