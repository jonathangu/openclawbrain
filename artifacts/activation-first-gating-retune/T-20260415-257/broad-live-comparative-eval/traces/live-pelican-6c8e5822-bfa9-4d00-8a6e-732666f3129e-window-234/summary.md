# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-234`
- winner mode: `graph_prior_only`
- trace hash: `sha256-80d3477f10050166bf08a79ad115cc0623875c77edbf3489b3449d2e77618193`
- fixture hash: `sha256-550621052f6f6f4dedd32e7dd1966df3bdae13f0842e74ffdcfed29aa308dfb9`
- score hash: `sha256-1a2fef9cf3864c202f930f0a7fefba139cb1deea2b5d4a4e380b745c21039e42`
- bundle hash: `sha256-5be5918ce2890d11b8629cf9a7f535c20272acfb08c0040c712bd23df50f1e77`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b053d8d2940510defb6223852e2cebf21b6ccd631a727caf5859e48b2c5a0baf |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6140a2a722707b6073f28b8d21472d1ef7ec9c75bcd4a8ef0eecc5597b943555 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dfbcd09282fd371934fe66537c0bcea07d77a465f926b56a6ec19af0b9cecf43 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f941fbaac7963c03a4758950e943d5c813f5eb7e64e44988a21b697fdaa7d6a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dbb9e4bb | sha256-b37ed3dcde77d869d62f22fdac3c2d09dce1a4435d180edad8bf34407d163807 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-dbb9e4bb | sha256-31ba61c6d7b5dbd91766bd15cb38287fe970e9e6a7b4e4d830001c8759f460f0 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d0de387e | sha256-896a15bc4f1e2ac902e0e1bd011ed9ffe41ccf0157cc9082fd1fd775c639ab75 |
