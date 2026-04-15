# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-076`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2475fe2a6c8f093359c7a9114559d8796f41e9f7cd1ec07db7813ffabc8da40c`
- fixture hash: `sha256-45ef9df8d6b446f8b1f701f41e5b1b2fc10fabaee25942a7631b331568aa14af`
- score hash: `sha256-7d697a491520a1373511fcf26f79d91754836e7a02b761ff76e416e3a7df9ff1`
- bundle hash: `sha256-d4c8448dffe449dc80cf101fd7e70d52b61f8794e1d1f104ad830306cfae694c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-18e80a3b5ca91ac2a6f8025eacf958282c07c535eea3cd785fd5adb98dc2f9c4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ddf564117480b9e888a581c117acb702e99e1bb6f278f6cec40b97d904c623ca |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c17cfe1b5a19124ed1887191c809a18920255daf12f6225907e4a9fe5076ba46 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7254c53db27d10677b37d6b738f9bce7de959dec4e1287c8d7664307a1ab3881 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-006db603 | sha256-e75007fb90f571fd7fcd09798357b85e6b5dd7fb8d114ba990079a1ca9f90ead |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-006db603 | sha256-1062904386a340bb9bd2cde4a766d0d30db817cfb8f9c42e8b91bc975188c0a0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-006db603 | sha256-e75007fb90f571fd7fcd09798357b85e6b5dd7fb8d114ba990079a1ca9f90ead |
