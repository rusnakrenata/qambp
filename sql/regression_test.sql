## Regression tests
with regression as (
select 
g.id, g.nr_of_nodes, g.edge_probability, 
g.pymetis_inter_edges, 
case when length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"0","")) <>
length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"1","")) then 0
else 1 end as metis_correct_partition,
g.kernighan_lin_inter_edges,
p.qa_inter_edges  as hybrid_inter_edges,
case when ((nr_of_nodes < 1000 and p.qa_inter_edges < 100000) or (nr_of_nodes >= 1000 and p.qa_inter_edges < 1000000 )) then 1 else 0 end as hybrid_solution_found,
case when p.qa_inter_edges > 900000 then abs(1000000-p.qa_inter_edges)
else nr_of_nodes/2 end as hybrid_partition,
g.lambda_est,
g.density,
g.duration_pymetis,
g.duration_keringham_lin,
p.duration_qa,
p.duration_full as duration_qa_full,
p.lambda_total 
from graphs g
inner join partitions p on p.graph_id = g.id
where description = "regression_KLPM"
and duration_qa is not null
# toto treba potom zakomentovat pre porovnanie ktory je lepsi
and ((nr_of_nodes < 1000 and p.qa_inter_edges < 100000) or (nr_of_nodes >= 1000 and p.qa_inter_edges < 1000000 ))
and nr_of_nodes <> 2500
and nr_of_nodes >= 100
)
select nr_of_nodes,
count(nr_of_nodes) as nr_of_graphs,
avg(edge_probability) as avg_edge_probability,
avg(density) as avg_density,
sum(metis_correct_partition ) as metis_correct,
sum(hybrid_solution_found) as hybrid_solution_found,
sum(case when metis_correct_partition = 1 and hybrid_solution_found = 1
	and hybrid_inter_edges = pymetis_inter_edges then 1 else 0 end)/count(nr_of_nodes) as hybrid_eq_pymetis,	
sum(case when (metis_correct_partition = 1 and hybrid_solution_found = 1 and hybrid_inter_edges > pymetis_inter_edges) or 
	(hybrid_solution_found = 0 and metis_correct_partition = 1) then 1 else 0 end)/count(nr_of_nodes) as pymetis_better,	
sum(case when (metis_correct_partition = 1 and hybrid_solution_found = 1 and hybrid_inter_edges < pymetis_inter_edges) or
	(hybrid_solution_found = 1 and metis_correct_partition = 0)then 1 else 0 end)/count(nr_of_nodes) as hybrid_better,
sum(case when hybrid_solution_found = 1
	and hybrid_inter_edges = pymetis_inter_edges then 1 else 0 end)/count(nr_of_nodes) as hybrid_eq_pymetis_wocorrect,	
sum(case when (hybrid_inter_edges > pymetis_inter_edges) or 
	(hybrid_solution_found = 0) then 1 else 0 end)/count(nr_of_nodes) as pymetis_better_wocorrect,	
sum(case when (hybrid_solution_found = 1 and hybrid_inter_edges < pymetis_inter_edges) then 1 else 0 end)/count(nr_of_nodes) as hybrid_better_wocorrect,	
avg(duration_pymetis) as avg_pymetis_duration,
avg(duration_keringham_lin) as avg_kl_duration,
avg(duration_qa) as avg_hybrid_duration,
avg(duration_qa_full-duration_qa) as avg_matrix_calc_duration,
avg(hybrid_inter_edges) avg_hybrid,
avg(pymetis_inter_edges) avg_pymetis,
abs(avg(hybrid_inter_edges) - avg(pymetis_inter_edges)) as absolute_difference,
100 * (avg(pymetis_inter_edges) - avg(hybrid_inter_edges)) / avg(hybrid_inter_edges) as percentage_difference,
ln(avg(pymetis_inter_edges)) - ln(avg(hybrid_inter_edges)) as log_difference,
(avg(pymetis_inter_edges) - avg(hybrid_inter_edges)) / nullif(stddev_samp(hybrid_inter_edges), 0) as cohen_d,
abs(avg(pymetis_inter_edges) - avg(hybrid_inter_edges)) / avg(hybrid_inter_edges) as relative_delta
from regression
group by nr_of_nodes
order by 1;