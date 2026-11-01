# Operator contracts and the live catalog

`OperatorRegistry` owns an optional `OperatorContract` for each registered
implementation. Existing operators without a declaration remain `UNDECLARED`.
The catalog never infers production readiness from registration or a schema.

A declaration binds the exact OperatorSpec hash, category (`basic`, `evolution`
or `evaluation`), named input/output schemas, a closed parameter schema,
modalities, effect and recovery semantics, and explicit limitations. Schemas
cannot resolve external references. Declaring retry behavior cannot expand the
registered implementation's retry authority.

The Controller validates declared parameters before recording a pending call or
invoking recovery/execution. Existing domain compilers and `verify_effect`
continue to validate provenance, artifacts, model/native receipts and quality.
Input/output schemas are descriptive metadata at this stage; this module does
not assert that every artifact has passed those schemas or independent tests.

The runtime Planner sees declarations only for operators authorized by its frozen
Goal. Its view omits executable argv, environment allowlists and filesystem roots.
Discovery adds no action kind, implementation or execution permission.

Declarations are copied into the registry and hash-bound in the frozen execution
contract. Changing, adding or removing one pauses execution before another
effect. Undeclared legacy registries retain the prior frozen payload shape.

Inspect an existing campaign without ticking it:

```sh
python -m dataevolver.harness.cli operators \
  --campaign-root "$CAMPAIGNS" --campaign "$CAMPAIGN"
```

The result reports current registration, declarations and frozen-contract drift;
it performs no inference, qualification, promotion or model health request.
Inspecting an older campaign through changed code can report contract drift.
Run that campaign only from its original frozen source.

The training trial declares a basic composite operation; registered probes
declare evaluation operations. Their configuration-bound sources and model and
resource authority remain in the frozen domain, and their port schemas cover
only the outer JSON shape. They do not claim general production qualification.

The next development-package gate must bind its request, candidate files,
implementation contract, independent protocol and protected source roots.
Neither this catalog nor an engineering pass authorizes candidate activation,
knowledge promotion or a method-improvement claim.
