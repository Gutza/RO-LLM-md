All files in this directory are samples of markdown files that are used to train and test the model.

The format of the files is:
```md
<<<

(original markdown text)

<<<

---

>>>

(fixed markdown text)

>>>
```

Note that all empty lines in the template above are mandatory; strictly speaking, the format is:
```<<<\n\n(original markdown text)\n\n<<<\n\n---\n\n>>>\n\n(fixed markdown text)\n\n>>>```

Windows line endings are also accepted (substitute `\n` with `\r\n` above).