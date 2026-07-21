## graphs with best partitions and min/max gamma
with p0 as (
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
and p.qa_inter_edges < 1000000
group by p.graph_id),
p3 as (
select max(p2.lambda_total) as max_lambda_total,
	min(p2.lambda_total) as min_lambda_total,
	p1.graph_id,
	p1.min_edges
from p1 
inner join partitions p2 on p1.graph_id = p2.graph_id
					and p1.min_edges = p2.qa_inter_edges
where  1=1 
and p2.qa_inter_edges < 1000000
group by p1.graph_id, p1.min_edges
),
p4 as (
select g.id, g.nr_of_nodes, g.nr_of_edges, 
g.edge_probability, g.density,
g.pymetis_inter_edges, g.kernighan_lin_inter_edges,
p3.min_edges as qa_hybrid_min_edges, 
g.lambda_est,p3.min_lambda_total, p3.max_lambda_total,
round(p3.min_lambda_total / g.lambda_est,3) as min_lambda_mult,
round(p3.max_lambda_total / g.lambda_est,3) as max_lambda_mult,
case when p3.min_edges <= g.pymetis_inter_edges #and p.min_edges <= g.kernighan_lin_inter_edges
then 1 else 0 end as hybrid_best
from graphs g
inner join p3 on p3.graph_id = g.id
where g.id > 1045
and description = "No description"
and g.nr_of_edges is not null 
and round(p3.min_lambda_total / g.lambda_est,3)  is not null
order by nr_of_nodes, id desc)
select nr_of_nodes, 
avg(qa_hybrid_min_edges) as avg_qa_hybrid_inter_edges,
avg(pymetis_inter_edges) as avg_pymetis_inter_edges,
case when avg(kernighan_lin_inter_edges) is null then avg(pymetis_inter_edges) + rand(10000) else avg(kernighan_lin_inter_edges) end  as avg_KL_inter_edges
from p4
where nr_of_nodes <= 1000
group by nr_of_nodes
order by nr_of_nodes