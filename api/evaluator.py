import sys

import database
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


def evaluate_pending_traces(project_name: str) -> dict:
    trace_ids: list[str] = []
    try:
        # 1. Fetch pending traces
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, prompt, response, contexts
                    FROM traces
                    WHERE eval_status = 'pending' AND project_name = %s
                    LIMIT 20
                    """,
                    (project_name,),
                )
                rows = cur.fetchall()

        # 2. Nothing to evaluate
        if not rows:
            return {"evaluated": 0, "errors": 0}

        trace_ids = [str(row[0]) for row in rows]
        prompts   = [row[1] for row in rows]
        responses = [row[2] for row in rows]
        contexts  = [row[3] if row[3] else [""] for row in rows]

        # 3. Mark as running so concurrent callers skip these rows
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE traces SET eval_status = 'running' WHERE id = ANY(%s)",
                    (trace_ids,),
                )
                conn.commit()

        # 4. Build RAGAS dataset
        dataset = Dataset.from_dict({
            "question": prompts,
            "answer":   responses,
            "contexts": contexts,
        })

        # 5. Run evaluation
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        df = result.to_pandas()

        # 6. Write scores back row-by-row
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                for i, trace_id in enumerate(trace_ids):
                    cur.execute(
                        """
                        UPDATE traces
                        SET faithfulness = %s,
                            relevancy    = %s,
                            eval_status  = 'complete'
                        WHERE id = %s
                        """,
                        (
                            float(df["faithfulness"].iloc[i]),
                            float(df["answer_relevancy"].iloc[i]),
                            trace_id,
                        ),
                    )
                conn.commit()

        return {"evaluated": len(trace_ids), "errors": 0}

    except Exception as exc:
        # 7. Reset all rows we touched back to failed
        print(f"[evaluator] error evaluating project={project_name!r}: {exc}", file=sys.stderr)
        if trace_ids:
            try:
                with database.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE traces SET eval_status = 'failed' WHERE id = ANY(%s)",
                            (trace_ids,),
                        )
                        conn.commit()
            except Exception as reset_exc:
                print(f"[evaluator] failed to reset trace status: {reset_exc}", file=sys.stderr)
        return {"evaluated": 0, "errors": len(trace_ids) if trace_ids else 1}
