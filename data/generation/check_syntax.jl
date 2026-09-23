function check(x)
    x isa Expr || return
    x.head in (:error, :incomplete) && error(x)
    foreach(check, x.args)
end
for file in ARGS
    check(Meta.parseall(read(file, String)))
    println(file)
end
