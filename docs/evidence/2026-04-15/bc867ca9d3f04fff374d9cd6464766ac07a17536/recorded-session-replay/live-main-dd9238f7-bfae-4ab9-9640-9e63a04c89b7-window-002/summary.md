# Recorded Session Replay Proof Bundle

- trace id: `live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f45e438e0f99f9d56b4e0ce3ef341383c4f2368651efec5583a2c7447c8a5e0`
- fixture hash: `sha256-24e221e1cec238f614a332fafbde124000574c7f4eca983f394d512d73646f16`
- score hash: `sha256-5896c729b19d31c7c3a56ef213e332fd25794489e5828c8c2123128953d11195`
- bundle hash: `sha256-465da4892a111d2fd4c002cfe663ad50ea1bbc6ac046785414ebdac83f6bf5ea`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5624a20b92c6a5c4c5d269dbfed46d621fb3009b7407cdb61d3d2abad216a892 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-77ff8a03ef52f4e8248f71077e1a833f54fbdb8680c60e0cd5ef48149ed7064e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5deec60698f8deee1d0f09c170e651cae334d4e07e58efb2fadba19500f586c0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-bf07b4f91e714fd6f38f728b0a33aca982112d4644e55d91c40d74f52106ba70 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce5a85ce | sha256-4fde4b1ae34e85ab15849bbe2065ec9a862336d0278bc1ca1cae12ff7310bd44 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ce5a85ce | sha256-f03cded7267e640a72d877e3f2f393bce32a997997cb884f38f6d1c12611778a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ce5a85ce | sha256-4fde4b1ae34e85ab15849bbe2065ec9a862336d0278bc1ca1cae12ff7310bd44 |
