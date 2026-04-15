# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6aeb45f46257078a73e31a3ca01fc811e5a3a9b2828328d1595fb41ae1cb1b87`
- fixture hash: `sha256-b90901422fe4620c22145acdd76fedd90d08a07ca2636957ff33166af8db8c6b`
- score hash: `sha256-fd861e3bd13deb6ba1b70c829cf2c7938017072e7d222a6f957164899e9d293c`
- bundle hash: `sha256-40b4acdc055d26a996f74d8fd298e6681d3a0dbb95145f3edd77b27826244b84`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9c00df70ff9e588e665c6961063a6f0105a883c9e9bd2b1d2f815eef1057f7d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-48273509bbf8ad878eb725c71bf905bd64c6f46a14d25f659a77a8ffa8fe971a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b06ab4eefea33732cceb851706cd8981cfbba58190b2c72ac383faaa4c74a1a5 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-075cd6f048e4c1f4ca9fe2b8a268836c00fb96070a56a7bfb41cbb3e48e38be9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ab9939bf | sha256-e7e32dfb5d5955f880dabb99c9d240d21e92aeee53d48d2706d699208b750855 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ab9939bf | sha256-88bae33a4ad487cb63b02385cf12f6ec48b0aba013618e4f547ccb093c03a647 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-ab9939bf | sha256-e7e32dfb5d5955f880dabb99c9d240d21e92aeee53d48d2706d699208b750855 |
