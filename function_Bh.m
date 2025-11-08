function Bh=function_Bh(value,num_HA_time,num_bands)
 temp=zeros(num_bands,1);
 for i=1:num_bands
    temp(i)=value(i)*sin(2*num_HA_time*pi*i/num_bands);
 end
Bh=2/num_bands*sum(temp);
