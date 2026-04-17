# Recorded Session Replay Proof Bundle

- trace id: `live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d851cdfc065d530ff6a05cd12aae1453cc6c5cc252f286f05c63b39f7b7ea103`
- fixture hash: `sha256-add4e01555ea0b700f89e1179ee076e863d3216d180ce57f607f066d853c468e`
- score hash: `sha256-8a64b0d7c7d7bc4c296c05f1592894148efd98545cec6fedfe08f4de5b35bb05`
- bundle hash: `sha256-5b1b9108a39a3b1064bece4d27ee2a973de1b1960d7cd8d1601c4d420eb80276`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16f2c51fd8866c40ce249b661c20fa44d3a586d3c45a550284b22e35e90bd83 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-96985f91ce24b62e014abb32ae2bf6581aaa47504ee33a0a4c709dc11aef9600 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c6c704798f4bec68ac15422f5be7c8836b4fedf946832f49f9fc70ecafbc6033 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b13f882c6521589128a2f248616ffd98ab470bee6a3e35395709095b392f8455 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c309d5ef | sha256-6814743b52f4c47b01dfd804f6864d7a8f57f02b3883fa2d0a7a027aee51c7c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c309d5ef | sha256-f77c7d95c962d5a4c0889a5b75b8c357dfff48b5658fbcc1b873c923023021dd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-343cf9ec | sha256-dd4c181d7454237819e0bd784ece726afe464f475abbbd7c9d797656fe17f719 |
