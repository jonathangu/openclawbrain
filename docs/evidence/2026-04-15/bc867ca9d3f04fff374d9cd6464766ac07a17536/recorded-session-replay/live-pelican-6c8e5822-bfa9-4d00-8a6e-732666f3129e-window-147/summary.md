# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147`
- winner mode: `graph_prior_only`
- trace hash: `sha256-12b53203712e88b756dee356041b3ddb0e18e328e1c8f8ade691064553599eca`
- fixture hash: `sha256-8ac6a4fe3950f0ed5cfb2e1b9bd9c7ad4d79faf9e22bb913250d8fa59920cf2e`
- score hash: `sha256-2e04c8e9e3b06dcfc1da6bd0e74d9ac6c8623dab86b01ac504283c3088c24273`
- bundle hash: `sha256-d654701df407eadd7af04d9b585f627de9e86ab6b45a9f979aee5f146a170bad`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55766afad53c9e202670418bdf755c0f71228a26fa5f954c36b74006ec3fe092 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9785e32a2cd1852238983ad4fc0aacd1cc04322927f745fd8d5f377e77ae24da |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7cba5effeb3c45e41dc2c000e81c857c5366c3da02e1efceb1b51ce31dedb2cb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-1464cdbd6076f23feed900c4b9b195c25d4f15850d804a25455d768097ed1bbf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cb137ac6 | sha256-670c0b04292658f977a53ffa4f8997a83f8ea629a14e977618bb7d9862d912a6 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cb137ac6 | sha256-06f276d0eec1d55a5b107a75988190c49c4560000b7c242d3f7682b831eff3c9 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cb137ac6 | sha256-670c0b04292658f977a53ffa4f8997a83f8ea629a14e977618bb7d9862d912a6 |
