# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-365dd958-3cda-46dc-a909-eca531206281-window-004`
- winner mode: `learned_route`
- trace hash: `sha256-414029967a4dfaeacf3048f9cc246c927617fc5206e50ba6c1c2944d9dd8d93c`
- fixture hash: `sha256-2f96d4d80b85de0482bdf816d900c02ecf0137642687879ce902112bb8056ccc`
- score hash: `sha256-c63686cd7144f1fb52680bfd02534b377f6369ff1e379f6a3b8eb60c6e0a7f1f`
- bundle hash: `sha256-db19b010f5ab4b5e0c27377a5bde8ece090bab1f064f38bcd2f15b65488d66e7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d02c67e07266eced41424ec8d8650df73f7c0173cd9e14609381c09dbbd89d1f |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c5cddfcc2d1707e3ebadc5046fe37878e344713b930fd24a66de940f2436cd61 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7be51faad159fbba2d14e30c034ddeb86c743244c861135f554e5bbdf2333124 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-d9710db6908d22727ddac562453471d92f36841ccedd1b9d6ff790fe525daabf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-ec06810b | sha256-ccf2d3526bb189a69dded7a942ed9299b6ced26abb6c29aedcfdae98e63a5618 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec06810b | sha256-8ed449087ef37bbea6d6542efc74658161ebcff93448bd9a5ea8786f24ea861d |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-93a2e1fe | sha256-bb5ec3f47da634948ecd19a5471d1776ad6b898c72080b4954cc70fd3688463e |
