function [INV_diag,iNV_sdiag]=inverse_3_times_3_blocks(d,sd)

if size(d,2)>1 || size(sd,2)>1
    error("Input of function inverse_3_times_3_blocks must be two column vectors.\n");
end

n=round(length(d)/3);
D11=d(1:n); D22=d(n+1:2*n); D33=d(2*n+1:end);
D12=sd(1:n);D13=sd(n+1:2*n);D23=sd(2*n+1:end);

DET=D11.*D22.*D33-(D11.*(D23.*conj(D23))+D22.*(D13.*conj(D13))+D33.*(D12.*conj(D12))) ...
        +2*real(D12.*D23.*conj(D13));%+D12.*D23.*conj(D13)+conj(D12).*conj(D23).*D13;

F11=(D22.*D33-D23.*conj(D23))./DET;
F22=(D11.*D33-D13.*conj(D13))./DET;
F33=(D11.*D22-D12.*conj(D12))./DET;

F12=(D13.*conj(D23)-D12.*D33)./DET;
F13=(D12.*D23-D13.*D22)./DET;
F23=(D13.*conj(D12)-D11.*D23)./DET;

if nargout==2
    INV_diag=[F11;F22;F33];
    iNV_sdiag=[F12;F13;F23];
else
    INV_diag=[F11;F22;F33;F12;F13;F23];
end

end