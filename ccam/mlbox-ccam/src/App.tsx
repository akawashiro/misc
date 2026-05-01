import { useMemo, useState } from 'react'
import './App.css'
import { compile } from './core/compiler'
import { formatProgram, formatValue, run } from './core/ccam'
import { parse } from './core/parser'

const sample = `eval (
  let cogen a = lift (6 * 7) in
    code (a + 8)
  end
)`

function App() {
  const [source, setSource] = useState(sample)

  const result = useMemo(() => {
    try {
      const ast = parse(source)
      const compiled = compile(ast)
      const executed = run(compiled.program)
      return { ast, compiled, executed, error: null }
    } catch (error) {
      return { ast: null, compiled: null, executed: null, error: error instanceof Error ? error.message : String(error) }
    }
  }, [source])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>ML^box to CCAM</h1>
          <p>Modal ML terms are parsed, compiled, and executed by a client-side CCAM simulator.</p>
        </div>
      </header>

      <section className="workspace">
        <section className="panel input-panel">
          <div className="panel-header">
            <h2>ML^box Input</h2>
          </div>
          <textarea
            value={source}
            onChange={(event) => setSource(event.target.value)}
            spellCheck={false}
            aria-label="ML box source"
          />
          <div className="syntax">
            <code>fn x =&gt; M</code>
            <code>code M</code>
            <code>lift M</code>
            <code>let cogen u = M in N end</code>
            <code>eval M</code>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Compile Trace</h2>
          </div>
          {result.error ? (
            <pre className="error">{result.error}</pre>
          ) : (
            <>
              <div className="summary">
                <span>CCAM program</span>
                <code>{formatProgram(result.compiled!.program)}</code>
              </div>
              <ol className="trace-list">
                {result.compiled!.log.map((line, index) => (
                  <li key={`${index}-${line}`}>
                    <code>{line}</code>
                  </li>
                ))}
              </ol>
            </>
          )}
        </section>

        <section className="panel run-panel">
          <div className="panel-header">
            <h2>CCAM Run</h2>
            {!result.error && <span>{result.executed!.transitions.length - 1} transitions</span>}
          </div>
          {result.error ? (
            <pre className="error">{result.error}</pre>
          ) : (
            <>
              <div className="summary">
                <span>result</span>
                <code>{formatValue(result.executed!.value)}</code>
              </div>
              <div className="transition-table">
                <div className="row heading">
                  <span>#</span>
                  <span>Stack</span>
                  <span>Program</span>
                </div>
                {result.executed!.transitions.map((transition) => (
                  <div className="row" key={transition.step}>
                    <span>{transition.step}</span>
                    <code>{transition.stack}</code>
                    <code>{transition.program}</code>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  )
}

export default App
