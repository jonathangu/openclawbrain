# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-065`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b834ef975b4127c9fa6cce4b12fc80a10ac45f8451c226003f814696763d3404`
- fixture hash: `sha256-2ef4285ae644d199abac210f4e94c99bfd3cbffd40a56868154cea15ccdb9a86`
- score hash: `sha256-3650a6d787c660f0eea11115fcf6ddc7c2d2ac23b8587d6ab96a8cc6531efa65`
- bundle hash: `sha256-7f9c1b58f9a261f8f8992e69f03a2bedf22cd5c951fdf77321aa12c0a349b9f9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f63fdc694dc12e3fd1585f1a9d1a8d63286b83507ef0e36210c868d071e50d26 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97e44df6632758fced83da959df46082f43609e64f148c686fe9665de980bddb |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54325123b03c35868697b2621e56ce98d1ed3ac802b0eedae03558b8f55715d2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e7481abca45904ad927ff911ac5db8e55d8d2c52bbc060ad215d380f456d87a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a6d96bf8 | sha256-71133f36362cb19eeb055b6b12c810393c0ccb36cc5ccf2ba10637817825f4c0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a6d96bf8 | sha256-732ec94293d0dd686eb10c6d424a3128d1f17c8d618adb7f2daf4ca28e12f8f5 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-04f3c4c3 | sha256-4aa19b22777f27ed4c3dd697891452d56755b9acaf041a081aa9db2aee59a05a |
