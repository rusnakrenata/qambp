-- Computational time comparison (Section 6.1 of the paper).
--
-- duration_pymetis        : PyMetis partitioning time
-- duration_qa             : solver time reported for the QA run
-- duration_qa_matrix_calc : QUBO matrix construction time, i.e. the classical
--                           overhead (duration_full - duration_qa)
--
-- NOTE: a CTE cannot be shared between statements, so the WITH clause is
-- repeated. Run whichever of the two queries you need.

-- Query 1: per-graph detail
with base as (
    select g.nr_of_nodes,
           g.id,
           g.duration_pymetis,
           g.duration_keringham_lin,
           g.description,
           p.duration_full - p.duration_qa as duration_qa_matrix_calc,
           p.duration_qa
    from graphs g
    inner join partitions p on g.id = p.graph_id
    where 1 = 1
      and g.description = 'regression_KLPM'
      and p.duration_full is not null
)
select *
from base
order by nr_of_nodes, id;


-- Query 2: averages per graph size (the figures quoted in Section 6.1)
with base as (
    select g.nr_of_nodes,
           g.id,
           g.duration_pymetis,
           g.duration_keringham_lin,
           g.description,
           p.duration_full - p.duration_qa as duration_qa_matrix_calc,
           p.duration_qa
    from graphs g
    inner join partitions p on g.id = p.graph_id
    where 1 = 1
      and g.description = 'regression_KLPM'
      and p.duration_full is not null
)
select nr_of_nodes,
       round(avg(duration_pymetis), 3)        as avg_duration_pymetis,
       round(avg(duration_keringham_lin), 3)  as avg_duration_kernighan_lin,
       round(avg(duration_qa_matrix_calc), 3) as avg_duration_qa_matrix,
       round(avg(duration_qa), 3)             as avg_duration_qa
from base
where nr_of_nodes >= 100
group by nr_of_nodes
order by nr_of_nodes;
