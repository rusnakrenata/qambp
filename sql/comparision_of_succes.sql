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
),
p0 as (
select g.id as graph_id
from partitions p
inner join graphs g on p.graph_id  = g.id
where nr_of_nodes in (100,200)
#and edge_probability in (0.1, 0.25, 0.5, 0.75)
and g.id > 1045
group by g.id
having sum(round(lambda_total/lambda_est,4)) >= 0.75
union all
select g.id
from partitions p
inner join graphs g on p.graph_id  = g.id
where nr_of_nodes in  (300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,4000)
#and edge_probability in (0.1, 0.25, 0.5, 0.75)
and g.id > 1045
group by g.id
having sum(round(lambda_total/lambda_est,4)) >= 0.195 
),
p1 as (
select min(p.qa_inter_edges) as min_edges,
	p.graph_id
from p0 
inner join partitions p on p.graph_id = p0.graph_id
where 1=1 
and p.comp_type = "hybrid"
group by p.graph_id),
lambda_calc as (
select
description,
g.id, g.nr_of_nodes, g.edge_probability, 
g.pymetis_inter_edges, 
p.qa_inter_edges  as hybrid_inter_edges,
case when ((nr_of_nodes < 1000 and p.qa_inter_edges < 100000) or (nr_of_nodes >= 1000 and p.qa_inter_edges < 1000000 )) then 1 else 0 end as hybrid_solution_found
from p1 
inner join graphs g on p1.graph_id = g.id
inner join partitions p on p1.graph_id = p.graph_id
					and p1.min_edges = p.qa_inter_edges
where description = "No description"
and g.id > 1045
# and duration_qa is not null
# toto treba potom zakomentovat pre porovnanie ktory je lepsi
# and ((nr_of_nodes < 1000 and p.qa_inter_edges < 100000) or (nr_of_nodes >= 1000 and p.qa_inter_edges < 1000000 ))
and nr_of_nodes <> 2500
and nr_of_nodes >= 100
)
select 
'GBR' as type,
nr_of_nodes,
count(nr_of_nodes) as nr_of_graphs,
sum(hybrid_solution_found) as hybrid_solution_found,
sum(case when (hybrid_solution_found = 1 and hybrid_inter_edges < pymetis_inter_edges) then 1 else 0 end)/count(nr_of_nodes)*100 as hybrid_better_pct,
abs(avg(hybrid_inter_edges) - avg(pymetis_inter_edges)) as absolute_difference,
100 * (avg(pymetis_inter_edges) - avg(hybrid_inter_edges)) / avg(hybrid_inter_edges) as percentage_difference
from regression
group by nr_of_nodes
UNION ALL
select 
'CALC' as type,
nr_of_nodes,
count(nr_of_nodes) as nr_of_graphs,
sum(hybrid_solution_found) as hybrid_solution_found,
sum(case when (hybrid_solution_found = 1 and hybrid_inter_edges < pymetis_inter_edges) then 1 else 0 end)/count(nr_of_nodes)*100 as hybrid_better_pct,
abs(avg(hybrid_inter_edges) - avg(pymetis_inter_edges)) as absolute_difference,
100 * (avg(pymetis_inter_edges) - avg(hybrid_inter_edges)) / avg(hybrid_inter_edges) as percentage_difference
from lambda_calc
group by nr_of_nodes
