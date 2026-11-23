-- 目标版本为 MySQL 8.0.42。先创建父表 law_versions，再创建引用它的 law_articles。

-- 每行登记一个官方法律文本版本，同一部法律的各版本共享 law_id。
CREATE TABLE law_versions (
    document_id       VARCHAR(128) NOT NULL,
    law_id            VARCHAR(128) NOT NULL,
    law_name          VARCHAR(256) NOT NULL,
    version_basis     TEXT NOT NULL,
    source_url        TEXT NOT NULL,
    promulgation_date DATE NOT NULL,
    effective_date    DATE NOT NULL,
    repeal_date       DATE NULL DEFAULT NULL,
    created_at        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    PRIMARY KEY (document_id),
    KEY idx_law_versions_law_effective (law_id, effective_date)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_bin
  ROW_FORMAT = DYNAMIC;

-- 每行保存一个版本的一条权威条文，保留完整正文及其 32 字节 SHA-256 摘要。
CREATE TABLE law_articles (
    kb_chunk_id VARCHAR(128) NOT NULL,
    document_id VARCHAR(128) NOT NULL,
    article_no  VARCHAR(64) NOT NULL,
    `text`      TEXT NOT NULL,
    text_sha256 VARBINARY(32) NOT NULL,
    source_url  TEXT NOT NULL,
    created_at  DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    PRIMARY KEY (kb_chunk_id),
    UNIQUE KEY uq_law_articles_document_article (document_id, article_no),
    CONSTRAINT fk_law_articles_version
        FOREIGN KEY (document_id) REFERENCES law_versions (document_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT chk_law_articles_text_sha256_length
        CHECK (OCTET_LENGTH(text_sha256) = 32)
) ENGINE = InnoDB
  DEFAULT CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_0900_bin
  ROW_FORMAT = DYNAMIC;
