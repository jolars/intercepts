function Pandoc(doc)
  if not doc.meta.abstract then
    error("The project metadata does not contain an abstract.")
  end

  local abstract = pandoc.utils.stringify(doc.meta.abstract)
    :gsub("%s+", " "):match("^%s*(.-)%s*$")
  if abstract == "" then
    error("The project abstract is empty.")
  end

  return pandoc.Pandoc({pandoc.Para({pandoc.Str(abstract)})})
end
