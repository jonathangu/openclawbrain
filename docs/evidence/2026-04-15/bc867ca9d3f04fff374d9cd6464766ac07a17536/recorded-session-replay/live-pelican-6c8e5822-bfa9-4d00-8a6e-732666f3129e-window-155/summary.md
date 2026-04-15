# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54c6a7f75aa98b64fa06de64444db8f288aa41bfaf9731cc070d54f577be960a`
- fixture hash: `sha256-899705aeb2321d03b6a0aee78d7cfb19ca0d976080db3e6a3f83db60267852fd`
- score hash: `sha256-503db094e3f6bd2c2850b3c67dff9172375f96452d6f3c2af1525e9afc15d1e0`
- bundle hash: `sha256-4f22179a73290a6f1297ad680a037d173da33b2e8f70256ce6cf227f0b6f7e3b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee198590c5a4a8c84e2f8fe36017d040fb15fc92428b4d0396417de634b42329 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cabaa481799c35ef1fcdf3926cdd284bfbe1eaab077b65e6705edc22daf0424e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fa688c024f6ac42344ab3f0f410414673797551fb520f032c3d171daa4296446 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c643a683141f4df2f8cfecec289bbcdeb857bfc0ffac3deec25c99faa60fe188 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5950aca5 | sha256-2f867e36cd6e143b12080aaac3e97671339400496ff14456118dcefb245cee82 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5950aca5 | sha256-304e3920fb0b14639629b706b4919c6d42e3ea88d928ba57d3e027c07a786417 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5950aca5 | sha256-2f867e36cd6e143b12080aaac3e97671339400496ff14456118dcefb245cee82 |
