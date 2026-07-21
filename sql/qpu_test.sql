with qpu_test as (
#QPU testovanie without KL
select 
g.id, g.nr_of_nodes, g.edge_probability, 
g.pymetis_inter_edges, 
case when length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"0","")) <>
length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"1","")) then 0
else 1 end as metis_correct_partition,
#length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"0","")) as count_metis_S0,
#length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"1","")) as count_metis_S1,
abs(nr_of_nodes/2-length(replace(replace(replace(replace(g.pymetis_partition_nodes, ", ",""),"[",""),"]",""),"1","")) ) as metis_partition,
g.kernighan_lin_inter_edges,
p.qa_inter_edges  as qpu_inter_edges,
case when p.qa_inter_edges < 1000000 then 1 else 0 end as qpu_solution_found,
case when p.qa_inter_edges > 900000 then abs(1000000-p.qa_inter_edges)
else nr_of_nodes/2 end as qpu_partition,
g.lambda_est,
g.density,
g.duration_pymetis,
g.duration_keringham_lin,
p.duration_qa as duration_qpu,
p.duration_full as duration_qpu_full,
p.lambda_total 
from graphs g
inner join 
(
select sub.graph_id, sub.qa_inter_edges, min(lambda_total) as min_lambda_total
from (
select graph_id, min(qa_inter_edges) as qa_inter_edges
from partitions 
where graph_id >=2000
group by graph_id) sub
inner join partitions p on p.graph_id = sub.graph_id
						and p.qa_inter_edges = sub.qa_inter_edges
group by sub.graph_id, sub.qa_inter_edges
)p_min
inner join partitions p on p.graph_id = p_min.graph_id
						and p.qa_inter_edges = p_min.qa_inter_edges 
						and p.lambda_total = p_min.min_lambda_total
on p.graph_id = g.id
where description = "qpu"
and duration_qa is not null
)
select nr_of_nodes, 
count(nr_of_nodes) as nr_of_graphs,
avg(edge_probability) as avg_edge_probability,
avg(density) as avg_density,
sum(metis_correct_partition ) as metis_correct,
sum(qpu_solution_found) as qpu_solution_found,
sum(case when metis_correct_partition = 1 and qpu_solution_found = 1
	and qpu_inter_edges = pymetis_inter_edges then 1 else 0 end)/count(nr_of_nodes) as qpu_eq_pymetis,	
sum(case when (metis_correct_partition = 1 and qpu_solution_found = 1 and qpu_inter_edges > pymetis_inter_edges) or 
	(qpu_solution_found = 0 and metis_correct_partition = 1) then 1 else 0 end)/count(nr_of_nodes) as pymetis_better,	
sum(case when (metis_correct_partition = 1 and qpu_solution_found = 1 and qpu_inter_edges < pymetis_inter_edges) or
	(qpu_solution_found = 1 and metis_correct_partition = 0)then 1 else 0 end)/count(nr_of_nodes) as qpu_better,
sum(case when qpu_solution_found = 1
	and qpu_inter_edges = pymetis_inter_edges then 1 else 0 end)/count(nr_of_nodes) as qpu_eq_pymetis_wocorrect,	
sum(case when (qpu_inter_edges > pymetis_inter_edges) or 
	(qpu_solution_found = 0) then 1 else 0 end)/count(nr_of_nodes) as pymetis_better_wocorrect,	
sum(case when (qpu_solution_found = 1 and qpu_inter_edges < pymetis_inter_edges) then 1 else 0 end)/count(nr_of_nodes) as qpu_better_wocorrect,	
avg(duration_pymetis) as avg_pymetis_duration,
avg(duration_keringham_lin) as avg_kl_duration,
avg(duration_qpu) as avg_qpu_duration,
avg(duration_qpu_full-duration_qpu) as avg_matrix_calc_duration
from qpu_test
group by nr_of_nodes
order by 1;