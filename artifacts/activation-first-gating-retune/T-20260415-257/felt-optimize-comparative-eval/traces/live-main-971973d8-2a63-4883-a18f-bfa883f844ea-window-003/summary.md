# Recorded Session Replay Proof Bundle

- trace id: `live-main-971973d8-2a63-4883-a18f-bfa883f844ea-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-caf66a8ae0f50cd2a6d3d3cfd5e877c0e6e80108a88f78a99bf9d0af8e14c2ff`
- fixture hash: `sha256-d0e5149f0ea8ad48a690b98cf321460b1ba9a083ea4dc63f8728a3baa728b105`
- score hash: `sha256-41477dc28c31a442f1d15d41b4e7405f5ea1237dbbe7ce28c6d55dd197a20458`
- bundle hash: `sha256-92b98fc1e6529897be7a026617f2445c37870d851bf7deadc5dcbdfef93e77b7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | vector_only | 70 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9082b7d02459eb2c821fe90896dfa510257befaff2be31c5d25dfe86c62c130d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-46d450ae76194779830d60cc1e83e424575dea5cad3bdb443217ca4dc4a0ecba |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-8ef630500810bb22ec8f102bc55b42d831d8adb7fd1e21962deda9fd47513298 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-02bdac1088756304adcd34ebebdecbd07fdf35c0e4fc97bc386d60ce7c084529 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-ec1fcf2c | sha256-142e00b1561a665714c73432b764de99df736eec0c67a655d0c51c4acee9a883 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-ec1fcf2c | sha256-be417c92c79e54ec04d4410f3f566695667a5d87f012f52d17100ebd43d34c7f |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-de81238d | sha256-fc8b1ded24386a9da2d8c0089ba8c213bfaef84cb452ab6f12a4043a4483b728 |
