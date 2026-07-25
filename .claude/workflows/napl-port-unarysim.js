export const meta = {
  name: 'napl-port-unarysim',
  description: 'Catalog every napl class, speed up + validate each (resumable via ledgers), and generate Verilog RTL for operations',
  whenToUse:
    'Run to sweep the napl software model: first port any UnarySim classes napl is still missing via the napl-gen-sim skill (so they flow through the rest of the run), inventory all classes (operation/module/metric/structure/algorithm) into reports/napl-port-unarysim-component-report.md, optimize each for CPU/GPU speed via the napl-opt-sim skill with an independent test-sweep gate, validate ports against UnarySim via the napl-validate-unarysim skill skipping anything already in reports/napl-validate-unarysim-report.md, and emit Verilog RTL via the napl-gen-rtl skill for operation classes not yet in reports/napl-gen-rtl-report.md then re-verify it via the napl-validate-sim-rtl skill (writing each RTL pipeline depth back to the class self.hw.pp_delay). The per-phase fan-out delegates to those single-kernel skills, and a serialized recorder per phase writes their results to the SAME per-skill reports under reports/ (the single source of truth ledger_status reads; each skill agent\'s own recording is overridden to avoid concurrent writes). Done-state is computed deterministically (ledger_status.py), not by eyeballing tables. Idempotent on rerun: Improve skips files unchanged since reports/napl-opt-sim-report.md (git-hash keyed); completed validations and verified/skipped RTL are skipped (failed RTL is retried); a class whose source Improve changed is re-validated. Validation disagreements are independently re-checked before recording. Scope a run with args {phases:["improve","validate","rtl"], subpackages:[...], classes:[...]}.',
  phases: [
    { title: 'GenSim', detail: 'scan UnarySim for classes napl lacks; port each missing one via napl-gen-sim skill (serialized); record' },
    { title: 'Discover', detail: 'scan src for classes, parse ledgers (ledger_status.py) + git hashes, write napl-port-unarysim-component-report.md' },
    { title: 'Improve', detail: 'one agent per changed file via napl-opt-sim skill; independent test-sweep gate; record post-edit hash' },
    { title: 'Validate', detail: 'one agent per not-yet-validated file via napl-validate-unarysim skill; re-check disagreements; serialized record' },
    { title: 'RTL', detail: 'one agent per operation class via napl-gen-rtl skill (skip verified/skipped); re-verify via napl-validate-sim-rtl skill; serialized record' },
    { title: 'Finalize', detail: 'sync self.hw.pp_delay from ledger + refresh napl-port-unarysim-component-report.md columns from the parser' },
    { title: 'Summary', detail: 'write napl-port-unarysim-summary-report.md reporting all subagent results' },
  ],
}

const REPO = '/Users/diwu/Projects/napl'
const SRC = `${REPO}/src/napl`
const SUBPACKAGES = ['operation', 'module', 'metric', 'structure', 'algorithm']
// Single source of truth: done-state lives in the per-skill reports under reports/, NOT in
// separate workflow ledgers. The workflow's serialized recorders write to these same reports
// (using the skills' own recorders) and ledger_status.py reads them back for idempotency.
const REPORTS = `${REPO}/reports`
const VALIDATION_LEDGER = `${REPORTS}/napl-validate-unarysim-report.md`   // napl-validate-unarysim skill
const RTL_LEDGER = `${REPORTS}/napl-gen-rtl-report.md`                     // napl-gen-rtl skill
const IMPROVE_LEDGER = `${REPORTS}/napl-opt-sim-report.md`                 // napl-opt-sim skill
const GENSIM_LEDGER = `${REPORTS}/napl-gen-sim-report.md`                  // napl-gen-sim skill
// The workflow's own outputs (class inventory, run summary, gap-analysis plan) also live under
// reports/ with the workflow-name prefix, matching the per-skill report naming.
const COMPONENTS = `${REPORTS}/napl-port-unarysim-component-report.md`
const SUMMARY = `${REPORTS}/napl-port-unarysim-summary-report.md`
// Each phase records with the SAME recorder its skill uses, so workflow rows and standalone
// skill rows share one schema per report.
const VALIDATE_RECORDER = `${REPO}/.claude/skills/napl-validate-unarysim/scripts/record_validation.py`
const GENRTL_RECORDER = `${REPO}/.claude/skills/napl-gen-rtl/scripts/record_gen_rtl.py`
const OPT_RECORDER = `${REPO}/.claude/skills/napl-opt-sim/scripts/record_opt.py`
const GENSIM_RECORDER = `${REPO}/.claude/skills/napl-gen-sim/scripts/record_gen_sim.py`
const MAPPING = `${REPO}/.claude/skills/napl-validate-unarysim/references/mapping.md`  // napl<->UnarySim name map
const RECORDER = VALIDATE_RECORDER
const RTL_RECORDER = GENRTL_RECORDER
const IMPROVE_RECORDER = OPT_RECORDER
const WF_SCRIPTS = `${REPO}/.claude/workflows/napl-port-unarysim-scripts`
// Deterministic done-detection parser: prints JSON of what is already done (validated module ids,
// rtl verified/skipped/failed class names, improve Source->git-hash map) by reading the reports,
// so done-detection is exact set membership in JS, not an agent eyeballing markdown tables.
const LEDGER_STATUS = `${WF_SCRIPTS}/ledger_status.py`
const STATUS_CMD =
  `conda run -n napl python ${LEDGER_STATUS} --validation ${VALIDATION_LEDGER} ` +
  `--rtl ${RTL_LEDGER} --improve ${IMPROVE_LEDGER}`

// ----------------------------------------------------------------------------
// Run scoping (args). {phases, subpackages, classes} - each optional; absent
// means "all". phases gates the Improve/Validate/RTL fan-outs.
// ----------------------------------------------------------------------------

const opts = args && typeof args === 'object' && !Array.isArray(args) ? args : {}
const phaseSel = Array.isArray(opts.phases) ? opts.phases : null
const subSel = Array.isArray(opts.subpackages) ? opts.subpackages : null
const clsSel = Array.isArray(opts.classes) ? opts.classes : null
const wantPhase = (p) => !phaseSel || phaseSel.includes(p)
const inScope = (c) =>
  (!subSel || subSel.includes(c.subpackage)) && (!clsSel || clsSel.includes(c.name))

// ----------------------------------------------------------------------------
// Helpers (shared so each phase stays consistent)
// ----------------------------------------------------------------------------

// Path relative to the src/napl root, used for compact agent labels (e.g. operation/mul_and.py).
const relPath = (f) => f.replace(SRC + '/', '')
// Path relative to the repo root (e.g. src/napl/sim/operation/mul_and.py) - the key form used in the
// improve ledger and in `git hash-object` output, so hashes line up across runs.
const repoRel = (f) => f.replace(REPO + '/', '')

// Group classes by their defining file. Per-file agents avoid the concurrent-
// edit / concurrent-validate hazards that per-class agents would hit on a
// shared file.
function groupByFile(classes) {
  const by = {}
  for (const c of classes) (by[c.file] ??= []).push(c)
  return by
}

// Every fan-out agent returns data and leaves ledger writes to a single
// recorder per phase - parallel writes to one markdown file would corrupt it.
const NO_LEDGER =
  'Do NOT write any ledger file yourself - concurrent agents would corrupt a shared file; ' +
  'the workflow records results centrally afterward.'

// Guarded, serialized ledger write: spawn one recorder agent for `items`, or
// no-op (with a log line) when there is nothing new.
async function recordLedger({ items, noun, label, phase, prompt }) {
  if (!items.length) {
    log(`No new ${noun} to record.`)
    return null
  }
  log(`Recording ${items.length} ${noun} to the ledger (serialized).`)
  return agent(prompt, { label, phase, schema: RECORD_SCHEMA })
}

// ----------------------------------------------------------------------------
// Schemas
// ----------------------------------------------------------------------------

// Discover returns the raw class inventory plus the verbatim ledger-status JSON and a
// File->git-hash map; the workflow computes every done-flag in JS by exact set membership,
// so no agent ever decides "already done" by reading a markdown table.
const DISCOVERY_SCHEMA = {
  type: 'object',
  required: ['classes', 'ledger_status', 'file_hashes'],
  properties: {
    classes: {
      type: 'array',
      items: {
        type: 'object',
        required: [
          'name', 'file', 'subpackage', 'description',
          'is_placeholder', 'is_autograd_function', 'unarysim_counterpart',
        ],
        properties: {
          name: { type: 'string', description: 'Class name' },
          file: { type: 'string', description: 'Absolute path to the file defining the class' },
          subpackage: { type: 'string', enum: SUBPACKAGES },
          description: { type: 'string', description: 'One-line description of what the class does' },
          is_placeholder: { type: 'boolean', description: 'True if the class is an empty/stub placeholder with no real logic' },
          is_autograd_function: {
            type: 'boolean',
            description: 'True if the class subclasses torch.autograd.Function (an STE/backward helper, not a napl_base module)',
          },
          unarysim_counterpart: {
            type: 'string',
            description: 'Best-guess UnarySim module/class name, or empty string if none likely exists',
          },
        },
      },
    },
    ledger_status: {
      type: 'object',
      description: 'Verbatim JSON printed by ledger_status.py (do not edit its values).',
      required: ['validated', 'rtl_verified', 'rtl_skipped', 'rtl_failed', 'improved'],
      properties: {
        validated: { type: 'array', items: { type: 'string' }, description: 'napl module ids already in the napl-validate-unarysim report' },
        rtl_verified: { type: 'array', items: { type: 'string' }, description: 'napl classes with a verified RTL row' },
        rtl_skipped: { type: 'array', items: { type: 'string' }, description: 'napl classes recorded as skipped (no sound gate-level mapping)' },
        rtl_failed: { type: 'array', items: { type: 'string' }, description: 'napl classes recorded as failed' },
        improved: { type: 'object', description: 'File (repo-relative) -> git hash recorded in the napl-opt-sim report', additionalProperties: { type: 'string' } },
      },
    },
    file_hashes: {
      type: 'object',
      description: 'Repo-relative source file -> its CURRENT `git hash-object` value, for every distinct class file.',
      additionalProperties: { type: 'string' },
    },
  },
}

const IMPROVE_SCHEMA = {
  type: 'object',
  required: ['file', 'changed', 'classes_touched', 'changes_summary', 'tests_run', 'tests_pass', 'content_hash'],
  properties: {
    file: { type: 'string' },
    changed: { type: 'boolean', description: 'Whether any code was modified' },
    classes_touched: { type: 'array', items: { type: 'string' } },
    changes_summary: { type: 'string', description: 'What was changed and the perf rationale (or why nothing changed)' },
    tests_run: { type: 'string', description: 'Exact test command(s) run' },
    tests_pass: { type: 'boolean' },
    content_hash: {
      type: 'string',
      description: "The file's `git hash-object <file>` value AFTER any edits - recorded so a later run skips this file while its content is unchanged.",
    },
  },
}

// Independent post-Improve gate: the sweep exits non-zero and lists the failing
// files, and the agent reports them alongside the printed output + log.
const SWEEP_SCHEMA = {
  type: 'object',
  required: ['files', 'passed', 'all_pass', 'failing', 'notes'],
  properties: {
    files: { type: 'number', description: 'Number of test files the sweep ran' },
    passed: { type: 'number', description: 'Number of test files that ran without failing' },
    all_pass: { type: 'boolean' },
    failing: { type: 'array', items: { type: 'string' }, description: 'Test files that errored / did not pass, with a one-line reason' },
    notes: { type: 'string' },
  },
}

// One Validate agent handles one FILE and returns one ledger row per class it
// validated. Row fields mirror record_validation.py's arguments exactly so the
// recorder can pass them straight through.
const VALIDATE_FILE_SCHEMA = {
  type: 'object',
  required: ['file', 'rows', 'notes'],
  properties: {
    file: { type: 'string' },
    rows: {
      type: 'array',
      items: {
        type: 'object',
        required: ['module', 'ref', 'bitexact', 'agreement', 'cpu', 'gpu', 'regimes', 'agree'],
        properties: {
          module: { type: 'string', description: 'napl module id, exactly <subpackage>.<name>, e.g. operation.mul_and' },
          ref: { type: 'string', description: 'UnarySim class, e.g. FSUMul' },
          bitexact: { type: 'string', description: 'e.g. "yes (diff 0)" or "no (weight RNG)"' },
          agreement: { type: 'string', description: 'tolerance/RMSE vs bound when not bit-exact' },
          cpu: { type: 'string', description: 'CPU runtime, e.g. "93.2 ms"' },
          gpu: { type: 'string', description: 'GPU/MPS runtime, e.g. "53.4 ms" or "n/a"' },
          regimes: { type: 'string', description: 'polarities / known-answers / workload shape' },
          agree: { type: ['boolean', 'null'], description: 'Whether napl and UnarySim agree within bound' },
        },
      },
    },
    notes: { type: 'string' },
  },
}

// Independent re-check of a single reported disagreement (agree=false). A false disagreement
// reads as "the port is broken", so each is re-validated from scratch before it is recorded.
const DISAGREE_VERIFY_SCHEMA = {
  type: 'object',
  required: ['module', 'agree', 'bitexact', 'agreement', 'notes'],
  properties: {
    module: { type: 'string', description: 'napl module id being re-checked, e.g. operation.mul_and' },
    agree: { type: ['boolean', 'null'], description: 'Re-checked verdict: whether napl and UnarySim agree within bound' },
    bitexact: { type: 'string', description: 're-checked bit-exactness, same wording as the validation row' },
    agreement: { type: 'string', description: 're-checked tolerance/RMSE vs bound' },
    notes: { type: 'string', description: 'why the original verdict held or was overturned' },
  },
}

const RTL_SCHEMA = {
  type: 'object',
  required: ['class', 'op', 'status', 'rtl_path', 'make_test', 'pipeline_delay', 'notes'],
  properties: {
    class: { type: 'string' },
    op: { type: 'string', description: 'The OP= name used for make test, e.g. mul_and' },
    status: { type: 'string', enum: ['generated', 'verified', 'skipped', 'failed'] },
    rtl_path: { type: 'string', description: 'Path to the generated op dir/file, or empty if skipped' },
    make_test: { type: 'string', description: 'Result of `make test OP=<op>` (PASS/FAIL/n-a)' },
    pipeline_delay: {
      type: ['number', 'null'],
      description: 'RTL pipeline latency in i_clk cycles from input to the corresponding output ' +
        '(0 = purely combinational); null if skipped. When input->output paths (or multiple outputs) ' +
        'have differing latencies, report the MINIMUM across them.',
    },
    notes: { type: 'string' },
  },
}

// Independent re-verification of generated RTL: one agent per op (re-runs make
// test for that op), so this is the single-op result schema.
const RTL_VERIFY_ONE_SCHEMA = {
  type: 'object',
  required: ['op', 'pass', 'notes'],
  properties: {
    op: { type: 'string' },
    pass: { type: 'boolean', description: 'True iff `make test OP=<op>` printed PASS on a full match' },
    notes: { type: 'string' },
  },
}

const DELAYS_SCHEMA = {
  type: 'object',
  required: ['applied', 'notes'],
  properties: {
    applied: { type: 'number', description: 'Number of classes whose self.hw.pp_delay was set/updated' },
    notes: { type: 'string' },
  },
}

const RECORD_SCHEMA = {
  type: 'object',
  required: ['written', 'ledger', 'notes'],
  properties: {
    written: { type: 'number', description: 'Number of rows written/appended' },
    ledger: { type: 'string' },
    notes: { type: 'string' },
  },
}

const FINALIZE_SCHEMA = {
  type: 'object',
  required: ['validated_yes', 'rtl_yes', 'notes'],
  properties: {
    validated_yes: { type: 'number', description: 'Rows whose Validated? column is now yes' },
    rtl_yes: { type: 'number', description: 'Rows whose RTL? column is now yes' },
    notes: { type: 'string' },
  },
}

// GenSim scan: every UnarySim class with no napl counterpart, classified for porting.
const GENSIM_SCAN_SCHEMA = {
  type: 'object',
  required: ['missing'],
  properties: {
    missing: {
      type: 'array',
      items: {
        type: 'object',
        required: ['unarysim_class', 'file', 'proposed_napl_name', 'subpackage', 'paradigm', 'skip_reason'],
        properties: {
          unarysim_class: { type: 'string', description: 'UnarySim class name, e.g. FSULinearPC' },
          file: { type: 'string', description: 'UnarySim source file (relative), e.g. kernel/linear.py' },
          proposed_napl_name: { type: 'string', description: 'napl name in napl style (lowercase), e.g. linear_pc' },
          subpackage: { type: 'string', enum: SUBPACKAGES },
          paradigm: { type: 'string', description: 'streaming | single-shot-binary | metric' },
          skip_reason: { type: 'string', description: 'empty for a real port; else why it is not a standalone napl class (e.g. backward-only autograd helper, RNG/bitstream generator -> codec)' },
        },
      },
    },
  },
}

// One GenSim port agent handles one class; fields mirror record_gen_sim.py's arguments.
// module_file/class_name are what the serialized wiring agent needs to export it.
const GENSIM_PORT_SCHEMA = {
  type: 'object',
  required: ['napl', 'unarysim', 'status', 'test', 'validated', 'module_file', 'class_name', 'notes'],
  properties: {
    napl: { type: 'string', description: 'napl class created, exactly <subpackage>.<name>' },
    unarysim: { type: 'string', description: 'UnarySim source class' },
    status: { type: 'string', enum: ['ported', 'skipped', 'failed'] },
    test: { type: 'string', description: 'test file created, or empty' },
    validated: { type: 'string', description: 'napl-vs-UnarySim agreement (bit-exact / within bound), or why not' },
    module_file: { type: 'string', description: 'new module file written (relative to repo, e.g. src/napl/sim/module/linear_pc.py), or empty if not ported' },
    class_name: { type: 'string', description: 'the Python class/symbol name to export from the module, e.g. linear_pc' },
    notes: { type: 'string' },
  },
}

// The single serialized wiring agent reports how it stitched the parallel ports in.
const GENSIM_WIRE_SCHEMA = {
  type: 'object',
  required: ['wired', 'import_ok', 'notes'],
  properties: {
    wired: { type: 'array', items: { type: 'string' }, description: 'napl classes exported from their subpackage __init__.py' },
    import_ok: { type: 'boolean', description: 'true if every newly-wired class imports cleanly via its package' },
    notes: { type: 'string', description: 'mapping.md updates, import failures, anything notable' },
  },
}

// ----------------------------------------------------------------------------
// Phase 0 - GenSim. Port UnarySim classes napl is still missing BEFORE Discover,
// so newly-ported classes flow through Improve/Validate/RTL in the SAME run. The
// gap is computed by scanning UnarySim vs napl, so it is idempotent: an already-
// ported class exists in napl and is no longer a gap. Ports edit shared files
// (subpackage __init__.py, mapping.md), so they run SERIALIZED, and the phase is a
// barrier before Discover.
// ----------------------------------------------------------------------------

let gensimPorted = []
let gensimSkipped = []
if (wantPhase('gensim')) {
  phase('GenSim')

  const scan = await agent(
    `Scan for UnarySim classes that napl does NOT yet have, so the workflow can port them. Invoke the ` +
      `\`napl-gen-sim\` skill (Skill tool) and execute its Step 1 DIRECTLY (you are the worker; do NOT ` +
      `dispatch a further subagent): read ${REPO}/.claude/skills/napl-validate-unarysim/references/experiment-plan.md ` +
      `(the port gap analysis) and references/mapping.md (the napl<->UnarySim name correspondence), list ` +
      `UnarySim's kernel/metric/stream classes, and cross-reference napl (${SRC}/) to find which have NO napl ` +
      `counterpart. Names differ (UnarySim FSUMul == napl mul_csg), so a class is a gap only if mapping.md, the ` +
      `experiment-plan, AND a search of src/napl all show no equivalent. For each gap class return ` +
      `{unarysim_class, file, proposed_napl_name (napl naming), subpackage, paradigm, skip_reason (empty for a ` +
      `real port; else why it is not a standalone napl class)}. Do NOT port anything yet; only scan.`,
    { label: 'gensim:scan', phase: 'GenSim', schema: GENSIM_SCAN_SCHEMA },
  )

  const gaps = (scan?.missing || []).filter((m) => !subSel || subSel.includes(m.subpackage))
  const toPort = gaps.filter((m) => !m.skip_reason)
  gensimSkipped = gaps.filter((m) => m.skip_reason)
  log(`GenSim: ${gaps.length} gap class(es) in scope; porting ${toPort.length}, skipping ${gensimSkipped.length} non-port(s).`)

  // Ports run in PARALLEL: each agent writes ONLY its own new module file + test and
  // validates by importing the class directly from its module path - it does NOT touch
  // the shared subpackage __init__.py or mapping.md (those would race across agents).
  // A single serialized wiring agent below then exports every new class and updates
  // mapping.md in one pass, so the shared files get exactly one writer.
  gensimPorted = (
    await parallel(
      toPort.map((m) => () =>
        agent(
          `Port the UnarySim class \`${m.unarysim_class}\` (UnarySim ${m.file}) into napl as ` +
            `\`${m.subpackage}.${m.proposed_napl_name}\` (paradigm: ${m.paradigm}). Invoke the \`napl-gen-sim\` skill ` +
            `(Skill tool) and execute its Steps 2-6 DIRECTLY (you are the worker; do NOT dispatch a further subagent): ` +
            `read the UnarySim source and reimplement in napl conventions under coding-discipline.\n\n` +
            `SCOPE - write ONLY your own NEW files; do NOT edit any shared file:\n` +
            `- Write the module file src/napl/${m.subpackage}/${m.proposed_napl_name}.py (the new class).\n` +
            `- Write the test tests/${m.subpackage}/test_${m.proposed_napl_name}.py.\n` +
            `- Validate it faithful against UnarySim (invoke napl-validate-unarysim, identical inputs). IMPORTANT: ` +
            `in the test, import the class DIRECTLY from its module ` +
            `(\`from napl.${m.subpackage}.${m.proposed_napl_name} import <ClassName>\`), NOT from the package ` +
            `(\`from napl.${m.subpackage} import ...\`), because the workflow has not wired it into ${m.subpackage}/__init__.py yet.\n` +
            `- Do NOT edit src/napl/${m.subpackage}/__init__.py and do NOT edit ${MAPPING}; a later serialized step does the wiring.\n\n` +
            `OVERRIDE the skill's Step 6 recorder - do NOT write reports/napl-gen-sim-report.md; the workflow records ` +
            `centrally. Return {napl:"${m.subpackage}.${m.proposed_napl_name}", unarysim:"${m.unarysim_class}", status, ` +
            `test, validated, module_file:"src/napl/${m.subpackage}/${m.proposed_napl_name}.py", class_name (the exact ` +
            `Python class/symbol to export), notes}.`,
          { label: `gensim:port:${m.proposed_napl_name}`, phase: 'GenSim', schema: GENSIM_PORT_SCHEMA },
        ),
      ),
    )
  ).filter(Boolean)

  // Serialized wiring: ONE agent stitches every parallel port into the shared files
  // (subpackage __init__.py exports + mapping.md) and smoke-checks the package imports,
  // so the new classes are importable before Discover and the downstream phases run.
  const portedOk = gensimPorted.filter((r) => r.status === 'ported' && r.module_file && r.class_name)
  if (portedOk.length) {
    const wired = await agent(
      `Wire these freshly-ported napl classes into their packages and the name map. Each was written as a ` +
        `standalone module but is NOT yet exported. For EACH class below:\n` +
        portedOk
          .map((r) => `- export \`${r.class_name}\` from ${r.module_file} in src/napl/${r.napl.split('.')[0]}/__init__.py (as napl \`${r.napl}\`)`)
          .join('\n') +
        `\n\nDo it carefully, respecting ${REPO}/CLAUDE.md: keep the existing import ORDER in each __init__.py ` +
        `(module/__init__ imports linear before conv) and keep operation imports lazy where the gotchas require it - ` +
        `do not introduce an import cycle. Then update ${MAPPING} with each new napl<->UnarySim correspondence ` +
        `(${portedOk.map((r) => `${r.napl} <-> ${r.unarysim}`).join(', ')}). FINALLY smoke-check every class imports ` +
        `via its package: \`conda run -n napl python -c "from napl.<subpackage> import <ClassName>"\` for each. ` +
        `Return {wired:[napl classes exported], import_ok, notes}.`,
      { label: 'gensim:wire', phase: 'GenSim', schema: GENSIM_WIRE_SCHEMA },
    )
    if (wired && !wired.import_ok) {
      log(`GenSim: wiring reported import problems - ${wired.notes}`)
    }
    log(`GenSim: wired ${wired?.wired?.length || 0} new class(es) into their packages and mapping.md.`)
  }

  await recordLedger({
    items: gensimPorted,
    noun: 'gen-sim port rows',
    label: 'record:napl-gen-sim-report.md',
    phase: 'GenSim',
    prompt:
      `Append the following port rows to ${GENSIM_LEDGER} using the napl-gen-sim recorder ${GENSIM_RECORDER} - run ` +
      `it ONCE PER ROW, in sequence (never in parallel), e.g.:\n` +
      `  conda run -n napl python ${GENSIM_RECORDER} --file ${GENSIM_LEDGER} --napl <napl> --unarysim <unarysim> ` +
      `--status <status> --test "<test>" --validated "<validated>" --notes "<notes>"\n\n` +
      `Rows (JSON):\n${JSON.stringify(gensimPorted, null, 2)}\n\n` +
      `The recorder stamps today's date, dedupes, and sorts by napl class. Return how many rows you wrote.`,
  })
} else {
  log('GenSim phase skipped (not in args.phases).')
}

// ----------------------------------------------------------------------------
// Phase 1 - Discover all classes, capture the ledger-status JSON + current file
// hashes, and write napl-port-unarysim-component-report.md. Always runs; it catalogs everything regardless
// of scope. Done-detection is computed in JS from the parser output (below), not
// by the agent reading markdown tables.
// ----------------------------------------------------------------------------

phase('Discover')

if (phaseSel || subSel || clsSel) {
  log(`Scoped run - phases: ${phaseSel ? phaseSel.join(',') : 'all'}; subpackages: ${subSel ? subSel.join(',') : 'all'}; classes: ${clsSel ? clsSel.join(',') : 'all'}.`)
}

const discovery = await agent(
  `Scan the napl source tree for ALL class definitions in these subpackages: ${SUBPACKAGES.join(', ')} ` +
    `under ${SRC}/. Use ripgrep (e.g. \`rg -n "^class " ${SRC}/<subpackage>\`) to find every \`class\` definition, ` +
    `then read enough of each to summarize it.\n\n` +
    `For each class capture: name; absolute file path; subpackage; a one-line description; whether it is a ` +
    `placeholder/stub (empty body, NotImplemented, or pure scaffold per CLAUDE.md - e.g. most of structure/ and ` +
    `algorithm/fft); whether it subclasses torch.autograd.Function (an STE/backward helper such as _round_ste_fn or ` +
    `the *STE functions backing fxp/hub/tlut layers - NOT a napl_base module); and the best-guess UnarySim ` +
    `counterpart module/class name (repo github.com/diwu1990/UnarySim) or empty string if none likely exists.\n\n` +
    `DONE-STATE (do NOT eyeball ledgers - produce machine output the workflow trusts verbatim):\n` +
    `1. From ${REPO} run this exact command and return its JSON unchanged as \`ledger_status\`:\n` +
    `   ${STATUS_CMD}\n` +
    `2. For the set of DISTINCT source files defining the classes you found, compute each file's git blob hash with ` +
    `\`git hash-object <file1> <file2> ...\` (run from ${REPO}; it prints one hash per file in argument order). Return ` +
    `\`file_hashes\` as a map of REPO-RELATIVE path (e.g. "src/napl/sim/operation/mul_and.py") -> hash.\n\n` +
    `Then WRITE ${COMPONENTS} (overwrite any existing file): a markdown document titled "# NAPL Components", one ` +
    `\`##\` section per subpackage, each a table with columns: Class | File (repo-relative) | Description | ` +
    `Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL?. Fill Validated? = "yes" iff ` +
    `\`<subpackage>.<class>\` is in ledger_status.validated; for an \`operation\` class fill RTL? = "yes" iff the class ` +
    `name is in ledger_status.rtl_verified, "skipped" iff in rtl_skipped, else "no"; for every non-operation class fill ` +
    `RTL? = "n/a (operation-only)". End with a one-line total count plus counts of autograd.Function helpers, ` +
    `already-validated, and already-RTL'd classes.\n\n` +
    `Return {classes, ledger_status, file_hashes}. Do NOT modify any source files - only write napl-port-unarysim-component-report.md.`,
  { label: 'discover+napl-port-unarysim-component-report.md', phase: 'Discover', schema: DISCOVERY_SCHEMA },
)

const allClasses = discovery.classes
const ledgerStatus = discovery.ledger_status
const fileHashes = discovery.file_hashes || {}

// Compute every done-flag in JS by exact set membership against the parser output.
const validatedSet = new Set(ledgerStatus.validated || [])
const rtlVerifiedSet = new Set(ledgerStatus.rtl_verified || [])
const rtlSkippedSet = new Set(ledgerStatus.rtl_skipped || [])
const improvedHashes = ledgerStatus.improved || {}
for (const c of allClasses) {
  c.validation_done = validatedSet.has(`${c.subpackage}.${c.name}`)
  c.rtl_done = rtlVerifiedSet.has(c.name)
  c.rtl_skipped = rtlSkippedSet.has(c.name)
  // improve_done: the file is in the napl-opt-sim report AND its recorded hash equals the current hash.
  const rel = repoRel(c.file)
  c.improve_done = !!improvedHashes[rel] && !!fileHashes[rel] && improvedHashes[rel] === fileHashes[rel]
}

// napl-port-unarysim-component-report.md catalogs everything; the fan-out skips placeholders and
// torch.autograd.Function STE helpers (backward shims, not napl_base modules),
// and respects the run's scope.
const realClasses = allClasses.filter(
  (c) => !c.is_placeholder && !c.is_autograd_function && inScope(c),
)
const autogradHelpers = allClasses.filter((c) => c.is_autograd_function)
log(
  `Discovered ${allClasses.length} classes. Fan-out base (in scope): ${realClasses.length} ` +
    `(dropped ${allClasses.filter((c) => c.is_placeholder).length} placeholders, ` +
    `${autogradHelpers.length} autograd.Function helpers: ${autogradHelpers.map((c) => c.name).join(', ') || 'none'}).`,
)

// ----------------------------------------------------------------------------
// Phase 2 - Improve implementations (CPU/GPU speedup), grouped BY FILE, then an
// INDEPENDENT test-sweep gate. Barrier after this phase: Validate and RTL read
// the improved code. A file Improve changed forces re-validation of its classes.
// ----------------------------------------------------------------------------

let improved = []
let revalidated = 0
let sweep = null
let improveSkipped = 0

if (wantPhase('improve')) {
  phase('Improve')

  // #1 - skip files already improved at their current content (napl-opt-sim report hash matches);
  // a changed file (or one never improved) falls through and is re-examined.
  const improveByFile = groupByFile(realClasses.filter((c) => !c.improve_done))
  const improveFiles = Object.keys(improveByFile)
  improveSkipped = new Set(realClasses.filter((c) => c.improve_done).map((c) => c.file)).size
  log(`Improving ${improveFiles.length} files (one agent per file); skipped ${improveSkipped} unchanged file(s) already recorded in the napl-opt-sim report.`)

  improved = (
    await parallel(
      improveFiles.map((f) => () =>
        agent(
          `Optimize the napl source file ${f} for CPU/GPU speed; it defines these classes: ` +
            `${improveByFile[f].map((c) => c.name).join(', ')}.\n\n` +
            `Use the project's \`napl-opt-sim\` skill (Skill tool): invoke it and execute its methodology DIRECTLY ` +
            `(you are the worker it describes - run its Steps, do NOT dispatch a further subagent). For each ` +
            `non-placeholder class in this file, apply the skill's optimize loop: baseline the timing, make ` +
            `behavior-preserving speedups under coding-discipline (heed its in-place vs non-in-place dtype-promotion ` +
            `gotcha), and gate on that class's tests/<subpackage>/test_<class>.py still passing on every device ` +
            `(CPU and any GPU/MPS).\n\n` +
            `TWO OVERRIDES for this batch context:\n` +
            `- Do NOT run the skill's Step 6 recorder (reports/napl-opt-sim-report.md); the workflow records the ` +
            `improve ledger centrally to avoid concurrent writes.\n` +
            `- Do NOT run the skill's full downstream test sweep; the workflow runs ONE independent sweep gate after ` +
            `all Improve agents finish, so this class's own test is enough here.\n\n` +
            `Hard constraints (also enforced by the skill): numerical outputs MUST NOT change; follow ` +
            `${REPO}/CLAUDE.md (self.stype/self.ntype dtypes; NEVER >>/<< on float tensors, use ` +
            `pow2_lshift/pow2_rshift; keep the lazy operation imports in __init__); do NOT touch placeholder/stub ` +
            `classes; if there is no safe speedup, change nothing and say so.\n\n` +
            `Report what you changed (set changed=true only if you edited the file), the perf rationale, the exact ` +
            `test command, and pass/fail. FINALLY, after any edits, run \`git hash-object ${f}\` (from ${REPO}) and ` +
            `return its value as content_hash - the workflow records it so a later run skips this file while its ` +
            `content is unchanged.`,
          { label: `improve:${relPath(f)}`, phase: 'Improve', schema: IMPROVE_SCHEMA },
        ).then((r) => (r ? { ...r, file: f } : null)),
      ),
    )
  ).filter(Boolean)

  // #1 - a file Improve changed may have altered an already-validated class's
  // behavior; clear its validation_done so Validate re-checks it.
  const changedFiles = new Set(improved.filter((r) => r.changed).map((r) => r.file))
  if (changedFiles.size) {
    for (const c of realClasses) {
      if (c.validation_done && changedFiles.has(c.file)) {
        c.validation_done = false
        revalidated++
      }
    }
    log(`Improve changed ${changedFiles.size} file(s); forcing re-validation of ${revalidated} previously-validated class(es).`)
  }

  // #2 - independent post-Improve gate. The sweep exits non-zero and lists the
  // failing files; per-file Improve agents can miss cross-file/downstream
  // breakage this catches.
  sweep = await agent(
    `Run the napl test sweep ONCE as an independent post-Improve gate: from ${REPO} run ` +
      `\`conda run -n napl python tests/sweep_test.py\`. It prints one "Running: <path>" line per test file, exits ` +
      `zero when all pass, and on any failure exits non-zero and prints each failing test file; ` +
      `tests/sweep_test.log holds the tracebacks (ignore benign diagnostic lines such as "largest error"). ` +
      `Report: files (test files run), passed (number that ran without failing), failing (the test files it ` +
      `listed as failed, each with a one-line reason from the log), and all_pass (true only if it exited zero). ` +
      `Do NOT modify any files.`,
    { label: 'gate:test-sweep', phase: 'Improve', schema: SWEEP_SCHEMA },
  )
  if (sweep && !sweep.all_pass) {
    log(`WARNING: post-Improve test sweep reported failures: ${(sweep.failing || []).join(', ') || 'see notes'}. ` +
      `Continuing (Validate vs UnarySim is an independent check); see napl-port-unarysim-summary-report.md.`)
  }

  // #1 - record the post-edit hash of every examined file so the next run can skip
  // unchanged files. The opt-sim report is per-kernel, so emit one row per (file, class)
  // carrying the file's content_hash; ledger_status builds the Source->Hash idempotency map
  // from the source+hash columns.
  const improveRows = improved
    .filter((r) => r.content_hash)
    .flatMap((r) =>
      (improveByFile[r.file] || []).map((c) => ({
        kernel: `${c.subpackage}.${c.name}`,
        source: repoRel(r.file),
        hash: r.content_hash,
        changed: r.changed ? 'yes' : 'no',
        summary: r.changes_summary || '',
      })),
    )
  await recordLedger({
    items: improveRows,
    noun: 'improve rows',
    label: 'record:napl-opt-sim-report.md',
    phase: 'Improve',
    prompt:
      `Append the following optimization rows to ${IMPROVE_LEDGER} using the napl-opt-sim recorder ` +
      `${IMPROVE_RECORDER} - run it ONCE PER ROW, in sequence (never in parallel), e.g.:\n` +
      `  conda run -n napl python ${IMPROVE_RECORDER} --file ${IMPROVE_LEDGER} --kernel <kernel> ` +
      `--source <source> --hash <hash> --changed <yes|no> --gate "(napl-port-unarysim Improve)" ` +
      `--summary "<summary>"\n\n` +
      `Rows (JSON):\n${JSON.stringify(improveRows, null, 2)}\n\n` +
      `The recorder stamps today's date, dedupes, and sorts by kernel. The source+hash columns are the ` +
      `idempotency key a later sweep reads to skip unchanged files. Return how many rows you wrote.`,
  })
} else {
  log('Improve phase skipped (not in args.phases).')
}

// ----------------------------------------------------------------------------
// Phase 3 - Validate against UnarySim. ONE AGENT PER FILE, for every un-validated
// (or re-validation-forced) class - the Validate agent (not Discover's guess)
// decides whether a UnarySim counterpart exists. Reported disagreements are
// independently re-checked before recording. A single recorder appends rows.
// ----------------------------------------------------------------------------

let validatedFiles = []
let validationRows = []
let disagreeRechecked = []

if (wantPhase('validate')) {
  phase('Validate')

  // #6 - do NOT pre-filter on Discover's best-guess counterpart (that drops classes
  // whose counterpart the discovery agent failed to name). Pass every un-validated
  // in-scope class; the skill omits any class that genuinely has no counterpart.
  const validateByFile = groupByFile(realClasses.filter((c) => !c.validation_done))
  const validateFiles = Object.keys(validateByFile)
  log(`Validating ${validateFiles.length} files with un-validated classes (rest already in the ledger).`)

  validatedFiles = (
    await parallel(
      validateFiles.map((f) => () => {
        const classes = validateByFile[f]
        const list = classes
          .map((c) => `${c.name} (subpackage ${c.subpackage}${c.unarysim_counterpart && c.unarysim_counterpart.trim() ? `, UnarySim ~${c.unarysim_counterpart}` : ', no counterpart guessed - find one if it exists'})`)
          .join('; ')
        return agent(
          `Validate the napl classes defined in ${f} against their UnarySim counterparts. Classes to validate: ${list}.\n\n` +
            `Invoke the \`napl-validate-unarysim\` skill (Skill tool) and follow its methodology for EACH class: read the ` +
            `UnarySim reference from the local clone at /Users/diwu/Projects/UnarySim (per the skill; the GitHub fetch ` +
            `is fallback-only), run both implementations on IDENTICAL inputs, and determine whether the numerical ` +
            `results agree within the appropriate bound (bit-exact where it holds, the SC/quant tolerance otherwise). ` +
            `Also measure the napl module's CPU and GPU(MPS) runtime for the workload (synchronize MPS before timing).\n\n` +
            `The counterpart hints above are best-guesses; confirm the real UnarySim class yourself. If a class ` +
            `genuinely has no UnarySim counterpart, omit it from the rows.\n\n` +
            `OVERRIDE the skill's Step 5 (record-the-validation): do NOT run record_validation.py - this workflow ` +
            `records centrally to avoid concurrent-write races. ${NO_LEDGER} INSTEAD return one row per validated ` +
            `class with the exact fields record_validation.py needs ` +
            `(module id EXACTLY "${classes[0].subpackage}.<name>", ref, bitexact, agreement, cpu, gpu, regimes, agree).`,
          { label: `validate:${relPath(f)}`, phase: 'Validate', schema: VALIDATE_FILE_SCHEMA },
        )
      }),
    )
  ).filter(Boolean)

  validationRows = validatedFiles.flatMap((r) => r.rows || [])

  // #3 - independently re-check every reported disagreement before recording it; a
  // false agree=false reads as "the port is broken". One fresh validator per module.
  const disagreements = validationRows.filter((r) => r.agree === false)
  if (disagreements.length) {
    log(`Re-checking ${disagreements.length} reported disagreement(s) with an independent validator before recording.`)
    disagreeRechecked = (
      await parallel(
        disagreements.map((row) => () =>
          agent(
            `INDEPENDENTLY re-validate the napl module \`${row.module}\` against its UnarySim counterpart ` +
              `\`${row.ref}\`. A prior run reported they DISAGREE (agree=false: bit-exact "${row.bitexact}", ` +
              `agreement "${row.agreement}"). Treat that as an unverified claim - verify it from scratch.\n\n` +
              `Invoke the \`napl-validate-unarysim\` skill (Skill tool), read the UnarySim reference from the local clone ` +
              `at /Users/diwu/Projects/UnarySim, run both implementations on IDENTICAL inputs, and decide whether they ` +
              `agree within the appropriate bound (SC ~1/sqrt(N) for streaming kernels, quant bound for binary-domain, ` +
              `bit-exact where it holds). Common false-disagreement causes: correlated RNG/Sobol dims, polarity ` +
              `mismatch, a too-tight tolerance, or a timestep/shape mismatch - rule these out.\n\n` +
              `${NO_LEDGER} Return {module:"${row.module}", agree (re-checked), bitexact, agreement, notes (why the ` +
              `original verdict held or was overturned)}.`,
            { label: `recheck:${row.module}`, phase: 'Validate', schema: DISAGREE_VERIFY_SCHEMA },
          ),
        ),
      )
    ).filter(Boolean)
    // Overwrite the recorded verdict with the re-checked one (keeps the disagreement
    // only if it survives a second independent validation).
    const byModule = Object.fromEntries(disagreeRechecked.map((v) => [v.module, v]))
    validationRows = validationRows.map((r) => {
      const v = r.agree === false && byModule[r.module]
      if (!v) return r
      const overturned = v.agree !== false
      return {
        ...r,
        agree: v.agree,
        bitexact: v.bitexact || r.bitexact,
        agreement: v.agreement || r.agreement,
        regimes: `${r.regimes}${r.regimes ? '; ' : ''}re-checked: ${overturned ? 'agreement confirmed' : 'disagreement confirmed'}`,
      }
    })
    const overturned = disagreeRechecked.filter((v) => v.agree !== false).map((v) => v.module)
    log(`Disagreement re-check: ${overturned.length} overturned (now agree): ${overturned.join(', ') || 'none'}; ` +
      `${disagreeRechecked.length - overturned.length} confirmed as genuine disagreements.`)
  }

  await recordLedger({
    items: validationRows,
    noun: 'validation rows',
    label: 'record:napl-validate-unarysim-report.md',
    phase: 'Validate',
    prompt:
      `Append the following validation rows to ${VALIDATION_LEDGER} using the recorder ${RECORDER} - run it ` +
      `ONCE PER ROW, in sequence (never in parallel). Pass --file so it targets the napl-port-unarysim ledger, e.g.:\n` +
      `  conda run -n napl python ${RECORDER} --file ${VALIDATION_LEDGER} --module <module> --ref <ref> ` +
      `--bitexact "<bitexact>" --agreement "<agreement>" --cpu "<cpu>" --gpu "<gpu>" --regimes "<regimes>"\n\n` +
      `Rows (JSON):\n${JSON.stringify(validationRows, null, 2)}\n\n` +
      `The recorder stamps today's date, dedupes, and sorts the ledger, so order does not matter. ` +
      `Return how many rows you wrote.`,
  })
} else {
  log('Validate phase skipped (not in args.phases).')
}

// ----------------------------------------------------------------------------
// Phase 4 - RTL. ONLY the `operation` subpackage, ONE AGENT PER CLASS, skipping
// classes already VERIFIED or recorded as SKIPPED in the RTL ledger (failed ops
// are retried). Generated RTL is INDEPENDENTLY re-verified (make test); the
// outcome of EVERY attempt (verified/skipped/failed) is recorded so by-design
// skips are not re-attempted forever. A single recorder appends rows.
// ----------------------------------------------------------------------------

let rtl = []
let rtlVerified = []
let rtlVerifyDropped = []
let rtlAttempted = 0

if (wantPhase('rtl')) {
  phase('RTL')

  const operationClasses = realClasses.filter((c) => c.subpackage === 'operation')
  // #2 - skip both verified (rtl_done) and previously-skipped (no sound gate-level
  // mapping) classes; a `failed` row is NOT skipped, so failures are retried.
  const toRtl = operationClasses.filter((c) => !c.rtl_done && !c.rtl_skipped)
  rtlAttempted = toRtl.length
  const rtlSkippedCount = operationClasses.filter((c) => c.rtl_skipped).length
  log(`Generating RTL for ${toRtl.length} operation classes (operation-only; ${operationClasses.length - toRtl.length - rtlSkippedCount} already verified, ${rtlSkippedCount} previously skipped).`)

  const rtlPrompt = (c) =>
    `Generate Verilog RTL for the napl operation class \`${c.name}\` (defined in ${c.file}). The op directory and ` +
    `\`make test\` OP name is \`${c.name}\` (i.e. ${SRC}/imp/${c.name}/).\n\n` +
    `Use the project's \`napl-gen-rtl\` skill (Skill tool): invoke it and execute its Steps DIRECTLY (you are the ` +
    `worker it describes - run its Steps, do NOT dispatch a further subagent). It emits the rtl/tb/gen under the op ` +
    `dir from the Python model, makes \`conda run -n napl make test OP=${c.name}\` PASS bit-exactly, and computes the ` +
    `pp_delay (min across paths).\n\n` +
    `TWO OVERRIDES for this batch context (so the workflow's central ledgers stay the source of truth and shared ` +
    `source files are not edited concurrently):\n` +
    `- Do NOT write \`self.hw.pp_delay\` to the Python class yourself (skip the skill's Step 5 file edit). Operation ` +
    `classes share files (e.g. compare.py), so the workflow's Finalize phase applies all pp_delay edits serially. ` +
    `RETURN the pp_delay as \`pipeline_delay\` instead.\n` +
    `- Do NOT run the skill's Step 6 recorder yourself; the workflow records the RTL outcome centrally (its serialized ` +
    `recorder writes the same reports/napl-gen-rtl-report.md, avoiding concurrent writes from the fan-out).\n\n` +
    `If this class is not a per-timestep streaming circuit (no natural gate-level mapping), follow the skill and ` +
    `return status "skipped" (pipeline_delay null) rather than forcing an unsound design.\n\n` +
    `Return {class, op, status, rtl_path, make_test, pipeline_delay, notes}. ${NO_LEDGER}`

  // #6 - group by op directory (== class name) so any classes targeting the
  // same dir run sequentially instead of racing on the same files.
  const rtlByOp = {}
  for (const c of toRtl) (rtlByOp[c.name] ??= []).push(c)
  const dupeOps = Object.entries(rtlByOp).filter(([, cs]) => cs.length > 1).map(([op]) => op)
  if (dupeOps.length) {
    log(`RTL: multiple classes target op dir(s) ${dupeOps.join(', ')}; running each such group sequentially to avoid a write collision.`)
  }

  rtl = (
    await parallel(
      Object.values(rtlByOp).map((group) => async () => {
        const out = []
        for (const c of group) {
          out.push(await agent(rtlPrompt(c), { label: `rtl:${c.name}`, phase: 'RTL', schema: RTL_SCHEMA }))
        }
        return out
      }),
    )
  )
    .flat()
    .filter(Boolean)

  // #3 - independently re-validate each op the agents claim passed, via the
  // napl-validate-sim-rtl skill (test-driven inputs + reset equivalence, a stronger
  // gate than a bare make-test re-run), before trusting/recording it.
  const rtlClaimed = rtl.filter((r) => r.status === 'generated' || r.status === 'verified')
  rtlVerified = rtlClaimed
  if (rtlClaimed.length) {
    // One agent per op, in parallel. `make test OP=<op>` namespaces gen/vec/build
    // under <op>/, and the skill edits only the per-op gen_<op>.py, so the per-op
    // re-verifications do not collide on shared files.
    const rtlVerifyPrompt = (op) =>
      `Independently re-validate the RTL just generated for op \`${op}\`, using the project's ` +
      `\`napl-validate-sim-rtl\` skill. Invoke the skill (Skill tool) and execute its Steps DIRECTLY (you are the ` +
      `worker - do NOT dispatch a further subagent): make \`gen_${op}.py\` derive its golden vectors from ` +
      `\`test_${op}.py\`'s inputs, then run \`conda run -n napl make test OP=${op}\` from ${SRC}/imp and ` +
      `require a bit-exact PASS (the skill also checks reset equivalence from t=0). This is stronger than a bare ` +
      `make-test re-run: it validates the RTL against the TEST's input streams, not just whatever vectors the ` +
      `generator emitted.\n\n` +
      `OVERRIDE the skill's Step 6 recorder - do NOT write reports/napl-validate-sim-rtl-report.md; the workflow ` +
      `records the RTL outcome centrally. Return {op, pass, notes} (pass = the skill's co-sim PASSed on a full match).`

    const verifyResults = (
      await parallel(
        rtlClaimed.map((r) => () =>
          agent(rtlVerifyPrompt(r.op), { label: `verify:rtl:${r.op}`, phase: 'RTL', schema: RTL_VERIFY_ONE_SCHEMA }),
        ),
      )
    ).filter(Boolean)
    const passed = new Set(verifyResults.filter((x) => x.pass).map((x) => x.op))
    rtlVerified = rtlClaimed.filter((r) => passed.has(r.op))
    rtlVerifyDropped = rtlClaimed.filter((r) => !passed.has(r.op)).map((r) => r.op)
    if (rtlVerifyDropped.length) {
      log(`RTL verify: ${rtlVerifyDropped.length} op(s) did NOT pass make test on independent re-run and were NOT recorded: ${rtlVerifyDropped.join(', ')}`)
    }
  }

  // #2/#5 - record the outcome of EVERY attempt via the napl-gen-rtl recorder (keyed by
  // class, last-write-wins): verified (passed re-verify), skipped (no sound mapping), or
  // failed (claimed but failed re-verify, or self-reported failed). Delay is recorded only
  // for verified ops; skipped ops stop being re-attempted on the next run.
  const verifiedOps = new Set(rtlVerified.map((r) => r.op))
  const rtlRecordRows = rtl.map((r) => {
    let status
    if (r.status === 'skipped') status = 'skipped'
    else if (r.status === 'failed') status = 'failed'
    else status = verifiedOps.has(r.op) ? 'verified' : 'failed' // generated/verified claim, gated on re-verify
    return {
      class: r.class,
      rtl: r.op,
      status,
      make_test: status === 'verified' ? 'PASS' : status === 'failed' ? 'FAIL' : 'n/a',
      pp_delay: status === 'verified' && typeof r.pipeline_delay === 'number' ? r.pipeline_delay : '',
      notes: [r.rtl_path, r.notes].filter(Boolean).join(' - ') || '',
    }
  })
  await recordLedger({
    items: rtlRecordRows,
    noun: 'RTL generation rows',
    label: 'record:napl-gen-rtl-report.md',
    phase: 'RTL',
    prompt:
      `Append the following RTL-generation rows to ${RTL_LEDGER} using the napl-gen-rtl recorder ${RTL_RECORDER} - ` +
      `run it ONCE PER ROW, in sequence (never in parallel), e.g.:\n` +
      `  conda run -n napl python ${RTL_RECORDER} --file ${RTL_LEDGER} --class <class> --rtl <rtl> ` +
      `--status <status> --make-test "<make_test>" --pp-delay "<pp_delay>" --notes "<notes>"\n` +
      `(omit --pp-delay, or pass an empty string, when the row's pp_delay is empty.)\n\n` +
      `Rows (JSON):\n${JSON.stringify(rtlRecordRows, null, 2)}\n\n` +
      `The recorder stamps today's date, dedupes, and sorts by class. Status verified/skipped/failed is the ` +
      `idempotency signal a later sweep reads. Return how many rows you wrote.`,
  })
} else {
  log('RTL phase skipped (not in args.phases).')
}

// ----------------------------------------------------------------------------
// Phase 5 - Finalize. (a) Sync self.hw.pp_delay for EVERY operation class in the
// napl-gen-rtl report from its pp_delay column (idempotent, complete across runs). RTL agents
// returned the delay instead of editing Python - operation classes share files
// (e.g. compare.py), so a single serialized agent applies all edits without
// racing. pp_delay lives in the class's `self.hw = hw_params(...)` contract (it
// supersedes the legacy scalar self.delay per CLAUDE.md). (b) Refresh
// napl-port-unarysim-component-report.md's Validated?/RTL? columns from the ledgers.
// ----------------------------------------------------------------------------

let delays = null
let finalize = null

if (wantPhase('validate') || wantPhase('rtl')) {
  phase('Finalize')

  const allOperationClasses = allClasses.filter(
    (c) => c.subpackage === 'operation' && !c.is_placeholder && !c.is_autograd_function,
  )
  const opFileByName = Object.fromEntries(allOperationClasses.map((c) => [c.name, c.file]))

  if (wantPhase('rtl') && (rtlVerified.length || allOperationClasses.some((c) => c.rtl_done)) && Object.keys(opFileByName).length) {
    delays = await agent(
      `Synchronize each operation class's \`self.hw.pp_delay\` with the RTL pipeline delay recorded in ${RTL_LEDGER} ` +
        `(the napl-gen-rtl report). Read that report; for EACH row take the "napl class" name and its "pp_delay (cyc)" ` +
        `value - this is the RTL pipeline delay (skip rows with a missing or non-numeric delay). Look up the class's ` +
        `source file in the map below, open it, and in that class's \`__init__\` - after the napl_base ` +
        `\`super().__init__(...)\` call - set the class's hardware contract so its pp_delay equals <delay>:\n` +
        `  - The hardware contract is \`self.hw = hw_params(pp_delay=<delay>)\` (hw_params is defined in napl.sim.base; ` +
        `see operation/mul_and.py and operation/shiftreg.py for the exact idiom).\n` +
        `  - If the class already has a \`self.hw = hw_params(...)\` line, update only its \`pp_delay=\` argument to ` +
        `<delay>, preserving any other arguments (e.g. \`timing=\`).\n` +
        `  - If the class still uses the LEGACY scalar \`self.delay = <n>\`, REPLACE that line with ` +
        `\`self.hw = hw_params(pp_delay=<delay>)\` (this migration supersedes self.delay per CLAUDE.md).\n` +
        `  - If the class has neither, ADD \`self.hw = hw_params(pp_delay=<delay>)\` right after super().__init__.\n` +
        `  - Ensure the file imports hw_params: the import must read \`from napl.sim.base import napl_base, hw_params\` ` +
        `(add \`hw_params\` to the existing napl_base import if it is missing). Do this once per file.\n` +
        `Change ONLY the hardware-delay assignment (and the import if needed) per class; do not alter any other ` +
        `behavior. Class -> file map (JSON):\n${JSON.stringify(opFileByName, null, 2)}\n\n` +
        `Return how many classes you updated.`,
      { label: 'sync:self.hw.pp_delay', phase: 'Finalize', schema: DELAYS_SCHEMA },
    )
  } else {
    log('No RTL delays to sync.')
  }

  finalize = await agent(
    `Refresh ONLY the "Validated?" and "RTL?" columns of every class row in ${COMPONENTS} so they reflect the ` +
      `current ledger state. Leave all other columns, rows, sections, and ordering exactly as they are.\n\n` +
      `Get the ledger state from the deterministic parser (do NOT eyeball the markdown ledgers). From ${REPO} run:\n` +
      `  ${STATUS_CMD}\n` +
      `It prints JSON {validated, rtl_verified, rtl_skipped, rtl_failed, improved}. Then for each class row match by ` +
      `EXACT set membership:\n` +
      `- Validated? = "yes" iff \`<subpackage>.<class>\` is in \`validated\`, else "no".\n` +
      `- RTL is auto-implemented for the \`operation\` subpackage only. For an operation class set RTL? = "yes" iff the ` +
      `class name is in \`rtl_verified\`, "skipped" iff in \`rtl_skipped\`, else "no". For every non-operation class set ` +
      `RTL? = "n/a (operation-only)".\n\n` +
      `After editing the table cells, update the trailing count summary line (already-validated and already-RTL'd ` +
      `counts) to match. Return how many rows now read yes in each column.`,
    { label: 'finalize:napl-port-unarysim-component-report.md', phase: 'Finalize', schema: FINALIZE_SCHEMA },
  )
} else {
  log('Finalize skipped (no Validate/RTL this run).')
}

// ----------------------------------------------------------------------------
// Phase 6 - Summary. Write a human-readable report of every subagent's result
// to napl-port-unarysim-summary-report.md (the structured run data the orchestrator already holds).
// ----------------------------------------------------------------------------

phase('Summary')

const report = {
  scope: { phases: phaseSel || 'all', subpackages: subSel || 'all', classes: clsSel || 'all' },
  gensim_ported: gensimPorted.map((r) => ({
    napl: r.napl, unarysim: r.unarysim, status: r.status, validated: r.validated,
  })),
  gensim_skipped: gensimSkipped.map((m) => ({ unarysim: m.unarysim_class, reason: m.skip_reason })),
  improve: improved.map((r) => ({
    file: relPath(r.file), changed: r.changed, classes: r.classes_touched,
    tests_pass: r.tests_pass, summary: r.changes_summary,
  })),
  improve_skipped_unchanged: improveSkipped,
  improve_gate: sweep ? { files: sweep.files, passed: sweep.passed, all_pass: sweep.all_pass, failing: sweep.failing } : null,
  revalidated_forced: revalidated,
  validate: validationRows.map((r) => ({
    module: r.module, ref: r.ref, bitexact: r.bitexact, agreement: r.agreement,
    cpu: r.cpu, gpu: r.gpu, agree: r.agree,
  })),
  disagreements_rechecked: disagreeRechecked.map((v) => ({
    module: v.module, agree_after_recheck: v.agree, overturned: v.agree !== false, notes: v.notes,
  })),
  rtl: rtl.map((r) => ({
    class: r.class, op: r.op, status: r.status, make_test: r.make_test,
    pipeline_delay: r.pipeline_delay, verified: rtlVerified.some((v) => v.op === r.op), notes: r.notes,
  })),
  rtl_verify_dropped: rtlVerifyDropped,
  delays_applied: delays?.applied ?? 0,
  component_columns: { validated_yes: finalize?.validated_yes ?? null, rtl_yes: finalize?.rtl_yes ?? null },
}

await agent(
  `Write ${SUMMARY} (overwrite): a readable markdown report of this napl-port-unarysim run, titled ` +
    `"# napl-port-unarysim run summary". Use the structured results below - do NOT re-run anything or inspect the repo; ` +
    `just format what is given.\n\n` +
    `Structure it as: a one-paragraph overview with headline counts (run scope; UnarySim classes newly ported into ` +
    `napl by GenSim and how many were skipped as non-ports; files improved/changed and how many ` +
    `unchanged files were skipped via the improve ledger; the post-Improve sweep gate result and any failing tests; ` +
    `classes force-re-validated; validations and any disagreements, noting how many disagreements were overturned vs ` +
    `confirmed on independent re-check; RTL generated/verified/dropped/skipped; self.hw.pp_delay values written); then a ` +
    `section per phase. Improve: a table of file | changed? | tests pass? | what changed, plus the count of unchanged ` +
    `files skipped and the independent sweep-gate result (files/passed/all_pass and any failing). Validate: a table of ` +
    `module | UnarySim ref | bit-exact | agreement | CPU | GPU, and call out any row with agree=false; if ` +
    `disagreements_rechecked is non-empty, add a short list of each re-checked module with overturned vs confirmed and ` +
    `the reason. RTL: a table of class | op | status | make test | independently verified? | pipeline delay (cycles) | ` +
    `notes, and call out any op in rtl_verify_dropped (claimed but failed re-verification) and any status=skipped op ` +
    `(recorded so it is not re-attempted). GenSim: a table of napl class | UnarySim source | status | validated, plus ` +
    `the list of skipped non-ports with their reason. Finalize: self.hw.pp_delay edits applied and the napl-port-unarysim-component-report.md column counts. Keep ` +
    `it concise and factual.\n\n` +
    `Results (JSON):\n${JSON.stringify(report, null, 2)}\n\n` +
    `Return one sentence confirming the file was written.`,
  { label: 'write:napl-port-unarysim-summary-report.md', phase: 'Summary' },
)

// ----------------------------------------------------------------------------
// Return value
// ----------------------------------------------------------------------------

return {
  scope: report.scope,
  summary_doc: SUMMARY,
  components_doc: COMPONENTS,
  validation_ledger: VALIDATION_LEDGER,
  rtl_ledger: RTL_LEDGER,
  gensim: {
    ported: gensimPorted.filter((r) => r.status === 'ported').map((r) => r.napl),
    failed: gensimPorted.filter((r) => r.status === 'failed').map((r) => r.napl),
    skipped: gensimSkipped.map((m) => m.unarysim_class),
  },
  total_classes: allClasses.length,
  fan_out_classes: realClasses.length,
  improve: {
    files: improved.length,
    changed: improved.filter((r) => r.changed).length,
    skipped_unchanged: improveSkipped,
    tests_failing: improved.filter((r) => !r.tests_pass).map((r) => r.file),
    revalidated_forced: revalidated,
    sweep_gate: sweep ? { passed: sweep.passed, files: sweep.files, all_pass: sweep.all_pass, failing: sweep.failing } : null,
  },
  validate: {
    files_run: validatedFiles.length,
    rows_recorded: validationRows.length,
    disagree: validationRows.filter((r) => r.agree === false).map((r) => r.module),
    disagreements_overturned: disagreeRechecked.filter((v) => v.agree !== false).map((v) => v.module),
  },
  rtl: {
    attempted: rtlAttempted,
    verified: rtlVerified.map((r) => r.class),
    verify_dropped: rtlVerifyDropped,
    failed: rtl.filter((r) => r.status === 'failed').map((r) => r.class),
    skipped: rtl.filter((r) => r.status === 'skipped').map((r) => r.class),
  },
  finalize: {
    validated_yes: finalize?.validated_yes ?? null,
    rtl_yes: finalize?.rtl_yes ?? null,
    delays_applied: delays?.applied ?? 0,
  },
}
