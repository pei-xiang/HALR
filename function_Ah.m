function Ah=function_Ah(value,num_HA_time,num_bands)
 temp=zeros(num_bands,1);
 for i=1:num_bands
    temp(i)=value(i)*cos(2*num_HA_time*pi*i/num_bands);
end
Ah=2/num_bands*sum(temp);
