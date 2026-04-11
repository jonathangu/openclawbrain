# OpenClawBrain 0.4.41

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.41`
- `@openclawbrain/cli@0.4.41`

## Why this release exists

`0.4.41` exists to make the strongest shipped routing truth visible and operator-verifiable.

Fresh homes should start from the approved cold-start prior. Retrains and promotions should stay rooted in that same live `route_fn` family instead of drifting onto an unrelated base. Before this cut, the host could still be healthy while `status --detailed` and `proof` rendered the live lineage as `retrain_lineage_not_visible` because the persisted traced-learning surface was too thin.

This release closes that visibility seam.

## What changed

- the packaged traced-learning bridge now derives `retrainLineage` from durable promotion truth when the persisted surface is thin
- `status --detailed` now reports visible lineage for the active learned pack when the promotion truth already exists in the brain store / retrain package
- `proof` now carries the same surfaced lineage truth so operator bundles can show what the learned route inherited from
- the public release/docs story now says plainly that existing homes keep their learned preferences on top while inheriting from the stronger generic cold-start prior underneath

## Operator truth

This is not a new install workflow.

The canonical lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

What changes is the honesty of the readout. When the host already has durable promotion truth, the operator surfaces now show the inherited-from-cold-start lineage instead of hiding it behind an `unknown` status.

## What success looks like

On a healthy host with durable promotion truth, the detailed status output now includes a visible lineage line like:

```text
lineage     status=visible prior=... candidate=... priorRooted=yes promotionValid=yes ...
```

That line is the practical operator check that the learned route is still rooted in the right prior family.

## Focused verification

- `node --test packages/cli/dist/test/traced-learning-bridge.test.js packages/cli/dist/test/status-learning-path.test.js`
- `npm --prefix packages/cli run release:verify`
- `openclawbrain proof --openclaw-home ~/.openclaw --skip-install --skip-restart`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If the host is already healthy, this release should preserve the same installed learning state while making its inherited cold-start lineage legible.
