## comment
## graphs where no partition was found
with p0 as (
select g.id as graph_id, count(density) as density_count
from partitions p
inner join graphs g on p.graph_id  = g.id
where nr_of_nodes in (100,200)
## and edge_probability in (0.1, 0.25, 0.5, 0.75)
and g.id > 1045
and description = "No description"
group by g.id
having sum(round(lambda_total/lambda_est,4)) >= 0.75
union all
select g.id as graph_id, count(density) as density_count
from partitions p
inner join graphs g on p.graph_id  = g.id
where nr_of_nodes in (10,30,50,80)
and edge_probability >=0.25#in (0.25, 0.35, 0.5, 0.75, 0.9)
and g.id > 1045
and description = "No description"
group by g.id
having sum(round(lambda_total/lambda_est,4)) >= 3.1
union all
select g.id, count(density) as density_count
from partitions p
inner join graphs g on p.graph_id  = g.id
where nr_of_nodes in  (300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000)
## and edge_probability in (0.1, 0.25, 0.5, 0.75)
and g.id > 1045
and description = "No description"
group by g.id
having sum(round(lambda_total/lambda_est,4)) >= 0.195 
),
p1 as (
select min(p.qa_inter_edges) as min_edges,
	p.graph_id,
	max(p0.density_count) as density_count
from p0 
inner join partitions p on p.graph_id = p0.graph_id
group by p.graph_id)
,p2 as (
select p1.*, g.nr_of_nodes, g.density, g.lambda_est, p.lambda_total
FROM  p1
inner join graphs g on g.id = p1.graph_id
inner join partitions p on p.qa_inter_edges = p1.min_edges and p.graph_id = g.id
## where p1.min_edges = 1000000
)
,p3 as (
select graph_id, nr_of_nodes, density, count(graph_id) as nr_of_partitions, round(count(graph_id)/density_count,3) as success_rate
from p2
group by graph_id, nr_of_nodes, density
)
,p4 as (
select nr_of_nodes, round(density,1) as density, avg(success_rate) as success_rate, sum(nr_of_partitions) as nr_of_graph_tested
from p3
## where nr_of_nodes >=100
group by nr_of_nodes, round(density,1)
)
select *
from p4
where nr_of_graph_tested >= 5 and nr_of_nodes <2500